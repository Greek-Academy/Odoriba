"""
スキャンした空間で、買おうか迷っている家具の候補をまとめて判定する。

    cd phase2
    python scan_candidates.py <scan.obj> --start X Y Z --goal X Y Z \\
        --item ソファA 180 85 80 --item 棚B 90 40 180
    python scan_candidates.py <scan.obj> --csv 候補.csv --any-side
    python scan_candidates.py --route --item 小 60 40 40 --item 大 200 90 90
    python scan_candidates.py --route --item 冷蔵庫 70 65 180 100 --carried-by diy

寸法は 幅 奥行 高さ(cm)。重さ(kg)が分かれば5つ目に書く(CSVなら5列目)。候補ごとに ○(入る見込み)/△(際どい)/×(入らない)と、
判定を決めた所の余裕を表にする。○は掃引だけで出した見込みで、断定ではない。「何cmまで入る」と1つの数字にしないのは、
幅・奥行・高さが取り合いになり、1つの上限では言えないため。

入る見込み・際どい候補には「何kgまでなら運べるか」も付ける。重さは傾けてよい
角度にだけ効くので、通すのに要る傾きから逆算できる(重さを聞かなくても出せる)。
重さを書いた候補は、その重さで傾けられる角度までで判定し、重さのせいで
入らないときは「重すぎる」と何kgまでなら入るかを出す。

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
LEGEND = "○ 入る見込み / △ 際どい(スキャンの誤差の内側) / × 入らない / ? 判定できない"
_ORDER = {"ng": 0, "close": 1, "ok": 2}

NOTES = (
    "掃引だけの判定で、回している途中の引っかかりは見ていない(○は甘めに出る)",
    "立てて回す運び方はまだ入っていないので、長い家具は厳しめに出る",
    "手すりがスキャンに写っていないと、実際より甘く出る",
    "経路の両端(家具の長さの半分+約60cm)は調べていない。目的地は最後の狭い所"
    "より先に置く",
    "家具の表面は粗い点で見ているので、大きい家具では手すりのような細い"
    "出っ張りを見逃すことがある",
    "「何kgまで」は、1人が支えられる重さを業者40kg・自分たち25kgと仮に置いた"
    "値(実測ではない)。商品ページの重さと見比べる目安にする",
)

# 1人が支えられる重さの設定(phase2_lstair.CARRIER_PRESETS)の、利用者に見せる名前
PRESET_TEXT = {"mover": "業者", "diy": "自分たち"}

# 余裕を決めた項を、利用者に見せる言葉にしたもの
TERM_TEXT = {"furniture": "家具と壁の間", "body": "運ぶ人と壁の間",
             "self": "運ぶ人と家具の間"}


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


@contextlib.contextmanager
def carried_weight(weight_kg, carried_by):
    """家具の重さと、誰が運ぶか(phase2_lstair のグローバル)を一時的に差し替える。

    weight_kg が None なら重さなし(傾きの上限は固定の値)で判定する。
    """
    orig = (L.FURN_WEIGHT_KG, L.CARRIER_PRESET)
    try:
        L.FURN_WEIGHT_KG = None if weight_kg is None else float(weight_kg)
        L.CARRIER_PRESET = carried_by
        yield
    finally:
        L.FURN_WEIGHT_KG, L.CARRIER_PRESET = orig


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


def _decide(space, bh):
    """運搬者ありの掃引の結果から、判定とそれを決めた位置を選ぶ。

    余裕が一番小さい位置だけで決めると甘くなる。直線部分と曲がる所で誤差の
    幅が違うので、たとえば直線で-4cm・曲がる所で-5cmなら、最小は曲がる所
    (±9cmの内側で「際どい」)だが、直線の-4cmは±3cmの外で「入らない」。
    掃引した全ての位置にその場所の誤差の幅を当て、一番悪い判定を採る。

    手が届く範囲(reach)の項は壁との距離ではない。それが余裕を決めていて
    値が正のときは、残りの項(壁や家具との距離)の最小値で判定する。
    戻り値: (判定, 位置s, 余裕cm, 曲がる所か, 効いた項の名前 or None)
    """
    cl = space.centerline
    limit = bh.get("limit") or {}
    term, terms = limit.get("term"), dict(limit.get("terms", {}))
    s_min, c_min = bh["s"], bh["capacity_cm"]
    at_min = cl.corner_at(s_min) is not None
    if bh["no_pose_s"] or (term == "reach" and c_min < 0):
        return "ng", s_min, c_min, at_min, term
    if term == "reach":
        terms.pop("reach")
        if terms:
            term = min(terms, key=terms.get)
            c_min = terms[term]
        else:
            term = None
    worst = None
    for s, c in bh["profile"]:
        if not np.isfinite(c):
            continue
        same = abs(s - s_min) < 1e-6
        c = c_min if same else c
        at = cl.corner_at(s) is not None
        verdict = classify(c, at)
        key = (_ORDER[verdict], c)
        if worst is None or key < worst[0]:
            worst = (key, verdict, s, c, at, term if same else None)
    if worst is None:
        return classify(c_min, at_min), s_min, c_min, at_min, term
    return worst[1:]


def _summarize(space, result):
    """judge の結果から、表に出す値だけを取り出す。"""
    bh, bf = result["bottleneck"], result["bottleneck_furniture_only"]
    verdict, s, c, at_corner, term = _decide(space, bh)
    limit = bh.get("limit") or {}
    swept = [r[0] for r in bh["profile"]]
    return {
        "verdict": verdict,
        "capacity_cm": c,
        "furniture_only_cm": bf["capacity_cm"],
        "s": s,
        "xy": np.asarray(space.centerline.xy_at(s), dtype=float).tolist(),
        "at_corner": at_corner,
        "scan_error_cm": scan_error(at_corner),
        "term": term,
        # 以下は余裕が一番小さい位置の値(判定を決めた位置と違うことがある)
        "min_capacity_cm": bh["capacity_cm"],
        "min_s": bh["s"],
        "min_term": limit.get("term"),
        "min_terms": limit.get("terms", {}),
        "contact_part": bh["contact"]["part"],
        "contact_xyz": bh["contact"]["xyz"],
        "no_pose_s": bh["no_pose_s"],
        "swept_s": [min(swept), max(swept)] if swept else None,
    }


def _needed_tilt(verdicts, verdict, max_tilt_deg):
    """その判定に届く一番小さい傾きの上限(度)。

    verdicts は [(傾きの上限, その上限での判定), ...]。max_tilt_deg(今の設定で
    許される傾き)以下の中から探す。見つからなければ max_tilt_deg を返す。
    """
    for t, v in sorted(verdicts):
        if t <= max_tilt_deg + 1e-9 and v == verdict:
            return t
    return max_tilt_deg


def _weight_info(space, result, row, num_carriers):
    """「何kgまで運べるか」を、傾きごとの掃引結果から出す。

    重さは傾けてよい角度にだけ効く(傾けるほど下側の人の負担が増える)。
    だから、今の判定に届く一番小さい傾きが分かれば、その傾きで下側の人が
    支えきれる重さが、判定が変わらない重さの上限になる。
    掃引の傾きは15度刻みなので、求まる傾きは実際より大きめ、上限の重さは
    小さめ(厳しめ)に出る。家具の寸法を差し替えた状態で呼ぶこと。

    重さを与えていて、もっと傾けられれば判定が良くなる場合(重さのせいで
    傾けられない)は weight_limited を真にし、良い方の判定(verdict_if_lighter)と
    それに届く重さの上限(weight_limit_if_lighter_kg)も入れる。
    """
    by_tilt = result.get("bottleneck_by_tilt")
    if not by_tilt:
        return {}

    def limits(tilt):
        return {name: float(L.weight_limit_kg(tilt, num_carriers, name))
                for name in L.CARRIER_PRESETS}

    verdicts = [(e["max_tilt_deg"], _decide(space, e)[0]) for e in by_tilt]
    max_tilt = result["bottleneck"]["max_tilt_deg"]
    info = {}
    if row["verdict"] in ("ok", "close"):
        tilt = _needed_tilt(verdicts, row["verdict"], max_tilt)
        info.update(needed_tilt_deg=tilt, weight_limit_kg=limits(tilt))
    if L.FURN_WEIGHT_KG is not None:
        best = max((v for _, v in verdicts), key=_ORDER.get)
        if _ORDER[best] > _ORDER[row["verdict"]]:
            tilt = _needed_tilt(verdicts, best, 90.0)
            info.update(weight_limited=True, verdict_if_lighter=best,
                        weight_limit_if_lighter_kg=limits(tilt))
    return info


def _too_heavy_row(num_carriers):
    """水平に持っても1人の負担が上限を超える候補の行(掃引はしていない)。"""
    load = max(L.carrier_loads_kg(L._pose(0.0, 0.0).as_quat(), num_carriers))
    return {"verdict": "ng", "term": "load", "too_heavy_flat": True,
            "capacity_cm": None, "no_pose_s": [], "at_corner": False,
            "flat_load_kg": float(load), "load_limit_kg": float(L.load_limit_kg())}


def _rank(row):
    """向きを比べるときの良さ(大きいほど良い)。"""
    c = row["capacity_cm"]
    return (_ORDER[row["verdict"]], not row["no_pose_s"], -np.inf if c is None else c)


def judge_candidates(space, candidates, num_carriers=2, any_side=False, log=print,
                     carried_by=None, on_row=None):
    """候補 [(名前, 幅, 奥行, 高さ) or (名前, 幅, 奥行, 高さ, 重さkg), ...] を1件ずつ判定する。

    space は中心線を作り終えた ScanSpace。戻り値は候補ごとの dict のリスト
    (入力と同じ順)。判定できなかった候補(経路が短すぎる等)は verdict が
    "unknown" で、理由が error に入る。1件の失敗で残りを止めない。

    重さがある候補は、下側の人の負担が1人の上限を超えない傾きまでで判定する
    (重さが無い候補は固定の傾き上限)。carried_by は誰が運ぶか
    (phase2_lstair.CARRIER_PRESETS の名前。既定は CARRIER_PRESET)で、
    1人の上限を決める。on_row を渡すと、1件終わるごとにその結果の dict で
    呼ぶ(画面に進み具合を出すため)。
    """
    carried_by = carried_by or L.CARRIER_PRESET
    rows = []
    for name, w, d, h, *rest in candidates:
        weight = float(rest[0]) if rest and rest[0] is not None else None
        best, errors = None, []
        for a, b, up in orientations(w, d, h, any_side):
            with furniture_dims(a, b, up), carried_weight(weight, carried_by):
                dims = [L.FURN_L, L.FURN_W, L.FURN_H]
                try:
                    result = SP.judge(space, num_carriers=num_carriers, rrt=False,
                                      tilt_detail=True, log=lambda *_: None)
                except SP.TooHeavyError:
                    row = _too_heavy_row(num_carriers)
                except ValueError as e:
                    errors.append(str(e))
                    continue
                else:
                    row = _summarize(space, result)
                    row.update(_weight_info(space, result, row, num_carriers))
            row["carried_as"] = dims
            if best is None or _rank(row) > _rank(best):
                best = row
        if best is None:
            best = {"verdict": "unknown", "error": errors[0] if errors else "向きが無い"}
        row = {"name": name, "size": [float(w), float(d), float(h)], **best}
        if weight is not None:
            row.update(weight_kg=weight, carried_by=carried_by)
        rows.append(row)
        log(format_row(row))
        if on_row:
            on_row(row)
    return rows


def shortage_text(row):
    """一番厳しい所の余裕を、利用者に見せる言葉にする。

    「あと何cm」と言えるのは壁との距離の項だけ。手が届く範囲の項が効いて
    いるときの値は別の量なので、cm では出さない(decisions.md 2026-11-23)。
    重さのせいで入らないときも同じで、何kgまでなら入るかを言う。
    """
    if row["verdict"] == "unknown":
        return row["error"]
    who = PRESET_TEXT.get(row.get("carried_by"))
    if row.get("too_heavy_flat"):
        return (f"重すぎる(水平に持っても1人{row['flat_load_kg']:.0f}kg、"
                f"{who}の上限は1人{row['load_limit_kg']:.0f}kg)")
    if row.get("weight_limited") and row["verdict"] == "ng":
        kg = row["weight_limit_if_lighter_kg"][row["carried_by"]]
        then = "入る見込み" if row["verdict_if_lighter"] == "ok" else "際どい"
        return f"重すぎる({who}が運ぶなら{int(kg)}kgまでで{then})"
    if row["no_pose_s"]:
        return "持てる姿勢が無い所がある"
    c = row["capacity_cm"]
    if row["term"] == "reach":
        return "手が届かない"
    where = "曲がる所" if row["at_corner"] else "直線部分"
    what = TERM_TEXT.get(row["term"])
    amount = f"余裕{c:.0f}cm" if c >= 0 else f"あと{-c:.0f}cm足りない"
    detail = "、".join(x for x in (where, what, f"誤差±{row['scan_error_cm']:.0f}cm") if x)
    return f"{amount}({detail})"


def row_parts(row):
    """1件の結果を、表の列ごとの言葉にする(端末の1行と画面の表で同じ文言を使う)。"""
    w, d, h = row["size"]
    size = f"{w:.0f}×{d:.0f}×{h:.0f}cm"
    if row.get("weight_kg") is not None:
        size += f" {row['weight_kg']:.0f}kg"
    extras = []
    carried = row.get("carried_as")
    if carried and carried[2] != h:
        extras.append(f"{carried[2]:.0f}cmの辺を上下にして運ぶ")
    limits = row.get("weight_limit_kg")
    if limits:
        # 切り捨てて出す(上限を少しでも超える数字を見せない)
        extras.append(" / ".join(f"{PRESET_TEXT[name]}なら{int(kg)}kgまで"
                                 for name, kg in limits.items()))
    return {"mark": MARKS[row["verdict"]], "name": row["name"], "size": size,
            "reason": shortage_text(row), "extras": extras}


def format_row(row):
    p = row_parts(row)
    text = f"{p['mark']} {p['name']}  {p['size']}  {p['reason']}"
    return text + "".join(f"  [{x}]" for x in p["extras"])


def parse_rows(records, header=True):
    """[名前, 幅, 奥行, 高さ(, 重さkg)] の文字列の並びを候補にする。

    (候補, 読めなかった行) を返す。読めない行は、黙って落とすと候補が
    減ったことに気づけないので、呼び出し側に返す。header が真なら、1行目が
    数値でないときだけ見出しとして飛ばす。重さ(5列目)は省いてよい。書いて
    あって数値でない行は、重さなしとして判定すると甘くなるので、読めない
    行にする。
    """
    items, skipped = [], []
    for i, rec in enumerate(records):
        if not any(x.strip() for x in rec):
            continue
        try:
            item = (rec[0].strip(), float(rec[1]), float(rec[2]), float(rec[3]))
            if len(rec) > 4 and rec[4].strip():
                item += (float(rec[4]),)
            items.append(item)
        except (ValueError, IndexError):
            if i > 0 or not header:
                skipped.append(",".join(rec))
    return items, skipped


def read_csv(path):
    """名前,幅,奥行,高さ[,重さkg] のCSVを読む。(候補, 読めなかった行) を返す。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return parse_rows(csv.reader(f))


def parse_text(text):
    """1行1候補の文字列(画面の入力欄)を読む。見出しは無いものとして扱う。"""
    return parse_rows(csv.reader(text.replace("、", ",").replace("，", ",").splitlines()), header=False)


def size_error(item):
    """候補の寸法・重さが正の有限の数でなければ、その理由を返す(よければ None)。"""
    name, *size = item
    if not all(np.isfinite(size)) or min(size) <= 0:
        return f"{name}: 寸法と重さは正の数(cm、kg)で指定する"
    return None


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
        if (start is None) != (goal is None):
            parser.error("--start と --goal は両方指定するか、両方省く")
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
    parser.add_argument("--item", nargs="+", action="append", default=[],
                        metavar="NAME W D H [KG]",
                        help="候補の名前と 幅 奥行 高さ(cm)、分かれば重さ(kg)。何回でも書ける")
    parser.add_argument("--csv", help="名前,幅,奥行,高さ[,重さkg] のCSV")
    parser.add_argument("--carriers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--carried-by", choices=sorted(L.CARRIER_PRESETS),
                        default=L.CARRIER_PRESET,
                        help="重さを書いた候補を誰が運ぶか。mover=業者(1人40kgまで)、"
                             "diy=自分たち(1人25kgまで)。どちらも仮の値")
    parser.add_argument("--any-side", action="store_true",
                        help="寝かせて運ぶ向きも試す(既定は高さを上下にしたまま)")
    parser.add_argument("--json", help="結果をJSONで書き出す先")
    args = parser.parse_args()

    try:
        items = [(n, *map(float, nums)) for n, *nums in args.item]
    except ValueError:
        parser.error("--item は 名前 幅 奥行 高さ [重さ](寸法と重さは数値)")
    if any(len(item) not in (4, 5) for item in items):
        parser.error("--item は 名前 幅 奥行 高さ の4つか、重さを足した5つ")
    if args.csv:
        from_csv, skipped = read_csv(args.csv)
        items += from_csv
        for line in skipped:
            print(f"警告: CSVのこの行は読めなかったので飛ばします: {line}")
    if not items:
        parser.error("候補が無い(--item か --csv で指定する)")
    for item in items:
        if size_error(item):
            parser.error(size_error(item))

    space = build_space(args, parser)
    print(f"経路の長さ {space.centerline.s_total:.0f}cm、運ぶ人 {args.carriers}人、"
          f"候補 {len(items)}件")
    if any(len(item) == 5 for item in items):
        print(f"重さを書いた候補は、{PRESET_TEXT[args.carried_by]}が運ぶ"
              f"(1人{L.CARRIER_PRESETS[args.carried_by]:.0f}kgまで)として判定")
    print(LEGEND)
    rows = judge_candidates(space, items, num_carriers=args.carriers,
                            any_side=args.any_side, carried_by=args.carried_by)
    print("\n注意:")
    for note in NOTES:
        print(f"  - {note}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"carriers": args.carriers, "any_side": args.any_side,
                       "carried_by": args.carried_by,
                       "load_limit_kg": dict(L.CARRIER_PRESETS),
                       "centerline_length": space.centerline.s_total,
                       "notes": list(NOTES), "candidates": rows},
                      f, ensure_ascii=False, indent=1)
        print(f"\n{args.json} に書き出しました")


if __name__ == "__main__":
    main()
