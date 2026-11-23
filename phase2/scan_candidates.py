"""
スキャンした空間で、買おうか迷っている家具の候補をまとめて判定する。

    cd phase2
    python scan_candidates.py <scan.obj> --start X Y Z --goal X Y Z \\
        --item ソファA 180 85 80 --item 棚B 90 40 180
    python scan_candidates.py <scan.obj> --csv 候補.csv --any-side
    python scan_candidates.py --route --item 小 60 40 40 --item 大 200 90 90

寸法は 幅 奥行 高さ(cm)。候補ごとに ○(入る)/△(際どい)/×(入らない)と、
一番厳しい所の余裕を表にする。「何cmまで入る」と1つの数字にしないのは、
幅・奥行・高さが取り合いになり、1つの上限では言えないため。

判定は scan_plan.judge の掃引(1件あたり十数秒〜1分)を寸法を変えて
繰り返すだけで、可搬性オラクルもプランナーも変えていない。
"""

import argparse
import contextlib
import csv
import json

import numpy as np

import phase2_lstair as L
import scan_plan as SP
import scan_space as SS

# スキャンの寸法がメジャーとどれだけずれるか(cm)。家全体のスキャンの採寸表
# (docs/decisions.md 2026-11-23)の値: 直線部分の壁から壁は3cm以内、曲がる所は
# 4〜9cmずれて再現しなかった。余裕がこの幅の内側なら、通るとも通らないとも
# 言えないので「際どい」にする。
SCAN_ERR_STRAIGHT = 3.0
SCAN_ERR_CORNER = 9.0

MARKS = {"ok": "○", "close": "△", "ng": "×", "unknown": "?"}

NOTES = (
    "掃引だけの判定で、回している途中の引っかかりは見ていない(○は甘めに出る)",
    "立てて回す運び方はまだ入っていないので、長い家具は厳しめに出る",
    "手すりがスキャンに写っていないと、実際より甘く出る",
)


def scan_error(at_corner):
    """その場所でのスキャンの誤差の幅(cm)。"""
    return SCAN_ERR_CORNER if at_corner else SCAN_ERR_STRAIGHT


def classify(capacity_cm, at_corner, no_pose=False, term=None):
    """余裕(cm)を "ok" / "close" / "ng" に分ける。

    - 持てる姿勢が1つも無い位置がある(no_pose) → "ng"
    - 手が届かないせいで余裕が負(term が "reach") → "ng"。この値は壁との
      距離ではないので、スキャンの誤差の幅は当てはめない
    - 余裕が誤差の幅以上 → "ok"、マイナス側に誤差の幅以上 → "ng"、
      その間 → "close"
    """
    if no_pose:
        return "ng"
    if term == "reach" and capacity_cm < 0:
        return "ng"
    err = scan_error(at_corner)
    if capacity_cm >= err:
        return "ok"
    if capacity_cm <= -err:
        return "ng"
    return "close"


@contextlib.contextmanager
def furniture_dims(a, b, height):
    """家具の寸法(phase2_lstair のグローバル)を一時的に差し替える。

    水平の2辺 a, b は大きい方を長辺にする(scan_picker の --furniture と同じ)。
    途中で例外が出ても元の寸法に戻す。
    """
    orig = (L.FURN_L, L.FURN_W, L.FURN_H)
    try:
        L.FURN_L, L.FURN_W = sorted((float(a), float(b)), reverse=True)
        L.FURN_H = float(height)
        yield
    finally:
        L.FURN_L, L.FURN_W, L.FURN_H = orig


def orientations(w, d, h, any_side=False):
    """試す向きの一覧 [(水平の辺, 水平の辺, 上下の辺), ...]。

    any_side が偽なら入力どおり(高さを上下に)。真なら、3辺のどれを上下に
    するかの3通り(同じ寸法の辺があれば重複は除く)。冷蔵庫のように寝かせ
    られない物があるので、既定では寝かせない。
    """
    if not any_side:
        return [(w, d, h)]
    seen, out = set(), []
    for a, b, up in ((w, d, h), (w, h, d), (d, h, w)):
        key = (max(a, b), min(a, b), up)
        if key not in seen:
            seen.add(key)
            out.append((a, b, up))
    return out


def _summarize(space, result):
    """judge の結果から、表に出す値だけを取り出す。"""
    bh, bf = result["bottleneck"], result["bottleneck_furniture_only"]
    at_corner = space.centerline.corner_at(bh["s"]) is not None
    no_pose = bool(bh["no_pose_s"])
    limit = bh.get("limit") or {}
    term = limit.get("term")
    return {
        "verdict": classify(bh["capacity_cm"], at_corner, no_pose, term),
        "capacity_cm": bh["capacity_cm"],
        "furniture_only_cm": bf["capacity_cm"],
        "s": bh["s"],
        "xy": bh["xy"],
        "at_corner": at_corner,
        "scan_error_cm": scan_error(at_corner),
        "contact_part": bh["contact"]["part"],
        "contact_xyz": bh["contact"]["xyz"],
        "term": term,
        "terms": limit.get("terms", {}),
        "no_pose_s": bh["no_pose_s"],
    }


def _rank(row):
    """向きを比べるときの良さ(大きいほど良い)。"""
    order = {"ok": 2, "close": 1, "ng": 0}
    return (order[row["verdict"]], not row["no_pose_s"], row["capacity_cm"])


def judge_candidates(space, candidates, num_carriers=2, any_side=False, log=print):
    """候補 [(名前, 幅, 奥行, 高さ), ...] を1件ずつ判定する。

    space は中心線を作り終えた ScanSpace。戻り値は候補ごとの dict のリスト
    (入力と同じ順)。判定できなかった候補(経路が短すぎる等)は verdict が
    "unknown" で、理由が error に入る。1件の失敗で残りを止めない。
    """
    rows = []
    for name, w, d, h in candidates:
        best, errors = None, []
        for a, b, up in orientations(w, d, h, any_side):
            with furniture_dims(a, b, up):
                dims = [L.FURN_L, L.FURN_W, L.FURN_H]
                try:
                    result = SP.judge(space, num_carriers=num_carriers, rrt=False,
                                      log=lambda *_: None)
                except ValueError as e:
                    errors.append(str(e))
                    continue
            row = _summarize(space, result)
            row["carried_as"] = dims
            if best is None or _rank(row) > _rank(best):
                best = row
        if best is None:
            best = {"verdict": "unknown", "error": errors[0] if errors else "向きが無い"}
        row = {"name": name, "size": [float(w), float(d), float(h)], **best}
        rows.append(row)
        log(format_row(row))
    return rows


def shortage_text(row):
    """一番厳しい所の余裕を、利用者に見せる言葉にする。

    「あと何cm」と言えるのは壁との距離の項だけ。手が届く範囲の項が効いて
    いるときの値は別の量なので、cm では出さない(decisions.md 2026-11-23)。
    """
    if row["verdict"] == "unknown":
        return row["error"]
    if row["no_pose_s"]:
        return "持てる姿勢が無い所がある"
    c = row["capacity_cm"]
    if row["term"] == "reach" and c < 0:
        return "手が届かない"
    where = "曲がる所" if row["at_corner"] else "直線部分"
    part = "運ぶ人" if row["contact_part"] == "carrier" else "家具"
    amount = f"余裕{c:.0f}cm" if c >= 0 else f"あと{-c:.0f}cm足りない"
    return f"{amount}({where}、{part}が当たる、誤差±{row['scan_error_cm']:.0f}cm)"


def format_row(row):
    w, d, h = row["size"]
    text = f"{MARKS[row['verdict']]} {row['name']}  {w:.0f}×{d:.0f}×{h:.0f}cm  {shortage_text(row)}"
    carried = row.get("carried_as")
    if carried and carried[2] != h:
        text += f"  [{carried[2]:.0f}cmの辺を上下にして運ぶ]"
    return text


def read_csv(path):
    """名前,幅,奥行,高さ のCSVを読む(数値でない行は見出しとして飛ばす)。"""
    items = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for rec in csv.reader(f):
            if len(rec) < 4:
                continue
            try:
                items.append((rec[0], float(rec[1]), float(rec[2]), float(rec[3])))
            except ValueError:
                continue
    return items


def build_space(args, parser):
    """メッシュを読み、空間と中心線を作る(scan_viewer と同じ手順)。"""
    if args.route:
        import mesh_env as M
        import route_env as R
        mesh = M.env_to_mesh(*R.build_route())
        mesh.invert()   # スキャンと同じ「面が自由空間側を向く」向きにする
        start, goal = R.START_XYZ, R.GOAL_XYZ
        seed = (R.HALL_W / 2, 40.0, 100.0)
    else:
        if not args.mesh:
            parser.error("mesh を指定する(または --route)")
        import scan_ingest as SI
        mesh, _ = SI.load_scan_zup(args.mesh)
        start, goal = args.start, args.goal
        if args.seed:
            seed = tuple(args.seed)
        elif start:
            seed = SS.seed_near(mesh, start)
        else:
            seed = SS.auto_seed(mesh)
    print("空間を作っています(約1分)...")
    space = SS.ScanSpace(mesh, seed)
    if start is None or goal is None:
        start, goal, walk = space.propose_endpoints()
        print(f"出発点・目的地を自動で提案: {np.round(start, 0).tolist()} -> "
              f"{np.round(goal, 0).tolist()} (歩いて{walk:.0f}cm)")
    space.build_centerline(start, goal)
    return space


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", nargs="?", help="スキャンのOBJ/PLY(--route なら不要)")
    parser.add_argument("--route", action="store_true", help="検証用の合成経路で試す")
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="自由空間の中の1点(既定は出発点の上から選ぶ)")
    parser.add_argument("--item", nargs=4, action="append", default=[],
                        metavar=("NAME", "W", "D", "H"),
                        help="候補の名前と 幅 奥行 高さ(cm)。何回でも書ける")
    parser.add_argument("--csv", help="名前,幅,奥行,高さ のCSV")
    parser.add_argument("--carriers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--any-side", action="store_true",
                        help="寝かせて運ぶ向きも試す(既定は高さを上下にしたまま)")
    parser.add_argument("--json", help="結果をJSONで書き出す先")
    args = parser.parse_args()

    try:
        items = [(n, float(w), float(d), float(h)) for n, w, d, h in args.item]
    except ValueError:
        parser.error("--item は 名前 幅 奥行 高さ(寸法は数値)")
    if args.csv:
        items += read_csv(args.csv)
    if not items:
        parser.error("候補が無い(--item か --csv で指定する)")

    space = build_space(args, parser)
    print(f"経路の長さ {space.centerline.s_total:.0f}cm、運ぶ人 {args.carriers}人、"
          f"候補 {len(items)}件")
    rows = judge_candidates(space, items, num_carriers=args.carriers,
                            any_side=args.any_side)
    print("\n注意:")
    for note in NOTES:
        print(f"  - {note}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"carriers": args.carriers, "any_side": args.any_side,
                       "centerline_length": space.centerline.s_total,
                       "notes": list(NOTES), "candidates": rows},
                      f, ensure_ascii=False, indent=1)
        print(f"\n{args.json} に書き出しました")


if __name__ == "__main__":
    main()
