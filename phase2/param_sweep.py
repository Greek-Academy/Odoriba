"""
寸法の掃引: 階段の型・幅・天井高と家具の組み合わせを大量に判定し、
「幅◯cmの階段なら長さ◯cmまで」の表を作る。家具店が公開している搬入の
目安と突き合わせるための材料(docs/decisions.md 参照)。

    python param_sweep.py                 # 全件(18コアで2時間ほど)。途中で止めても続きから再開する
    python param_sweep.py --limit 8       # 動作確認用に先頭8件だけ
    python param_sweep.py --workers 6     # 並列数を指定
    python param_sweep.py --table         # 保存済みの結果から表と自己矛盾チェックだけ出す

出力: results/param_sweep.jsonl (1行1件。コミットする)

判定の意味(いつもの注意):
  - RRTは回さず、ボトルネック掃引(中心線上の各点で、粗い姿勢グリッドの
    中の最良の余裕)だけを使う。余裕が負なら「試した姿勢のどれでも足りない」。
    余裕が正でも、回しきる経路があるとは限らない(必要条件であって十分条件ではない)
  - 踊り場は階段の幅と同じ一辺の正方形(各環境モジュールの前提)。
    踊り場だけ広い階段はこの掃引に含まれない
  - 重さは与えない(傾き上限55度固定)
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 並列ワーカーの中でnumpyがさらにスレッドを増やさないようにする(import numpy より前)
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "param_sweep.jsonl")

STAIR_TYPES = ("lstair", "ustair", "winder")
TYPE_LABELS = {"lstair": "L字(踊り場)", "ustair": "折り返し", "winder": "回り階段"}
# 建築基準法の住宅の階段は幅75cm以上。戸建てで多い範囲を5cm刻み、広い側は粗く
WIDTHS = (75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0, 120.0)
# 各フライト最上段の踏み面から天井までの高さ。220を先に回す
CEILS = (220.0, 200.0, 240.0)

# 長さを振る箱(断面45x60cm)。「何cmまで通るか」の境目を取るため
BOX_LENGTHS = (60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0)
BOX_W, BOX_H = 45.0, 60.0


def furniture_list():
    """(名前, L, W, H) の一覧。箱の系列 + catalog.py の定番家具。"""
    import catalog
    items = [(f"箱{l:.0f}", l, BOX_W, BOX_H) for l in BOX_LENGTHS]
    items += [(name, float(l), float(w), float(h)) for name, l, w, h, _ in catalog.CATALOG]
    return items


def all_cases():
    """判定する全ケース。天井高220を先に、その中では型→幅→家具の順。"""
    furn = furniture_list()
    return [{"type": t, "width": w, "ceil": c, "name": n, "L": l, "W": fw, "H": fh}
            for c in CEILS for t in STAIR_TYPES for w in WIDTHS for n, l, fw, fh in furn]


def case_key(c):
    return f"{c['type']}|{c['width']:.0f}|{c['ceil']:.0f}|{c['L']:.0f}x{c['W']:.0f}x{c['H']:.0f}"


def _env(stair_type):
    """型の名前から (環境モジュール, 環境を作る関数, 場所の名前を返す関数)。"""
    import phase2_lstair as L
    if stair_type == "lstair":
        return L, L.build_lstairs, L._where_on_skeleton
    if stair_type == "ustair":
        import phase2_ustair as U
        return U, U.build_ustairs, U.where_label
    import phase2_winder as W
    return W, W.build_winder, W.where_label


def run_case(c):
    """1ケースを判定する(ワーカープロセスで実行)。家具だけ／運搬者2人の
    それぞれについて、掃引の最小余裕(cm)とその場所を返す。"""
    import phase2_lstair as L
    t0 = time.time()
    mod, build, where = _env(c["type"])
    # 家具の寸法は phase2_lstair のグローバルを全環境が共有している
    L.FURN_L, L.FURN_W, L.FURN_H = c["L"], c["W"], c["H"]
    mod.configure(width=c["width"], ceil_clear=c["ceil"])
    outer, obstacles = build()
    out = dict(c, key=case_key(c))
    for with_human, tag in ((False, "solo"), (True, "carried")):
        sweep = mod.sweep_capacity(outer, obstacles, num_carriers=2, with_human=with_human)
        finite = [(s, cap) for s, cap, _ in sweep if np.isfinite(cap)]
        if not finite:
            # 運搬者の立つ床がどこにもない、など。判定不能として残す
            out[tag], out[tag + "_s"], out[tag + "_where"] = None, None, None
            continue
        s_min, c_min = min(finite, key=lambda r: r[1])
        out[tag] = round(float(c_min), 2)
        out[tag + "_s"] = round(float(s_min), 1)
        out[tag + "_where"] = where(s_min)
    out["time_s"] = round(time.time() - t0, 1)
    return out


def load_results(path=OUT_PATH):
    if not os.path.exists(path):
        return {}
    res = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                res[r["key"]] = r
    return res


# ---- 集計 ----

def max_box_length(res, stair_type, width, ceil, tag):
    """箱の系列で、余裕が0以上になる最大の長さ(20cm刻み)。全部通らなければ None、
    未計算の長さがあれば "?" を付けた文字列を返す。"""
    best, missing = None, False
    for l in BOX_LENGTHS:
        r = res.get(case_key({"type": stair_type, "width": width, "ceil": ceil,
                              "L": l, "W": BOX_W, "H": BOX_H}))
        if r is None:
            missing = True
        elif r[tag] is not None and r[tag] >= 0:
            best = l
    if best is None:
        return "?" if missing else f"<{BOX_LENGTHS[0]:.0f}"
    return f"{best:.0f}" + ("?" if missing else "")


def print_tables(res):
    print(f"\n保存済み {len(res)} 件 / 全 {len(all_cases())} 件")
    for ceil in CEILS:
        print(f"\n### 箱(断面{BOX_W:.0f}x{BOX_H:.0f}cm)が通る最大の長さ(cm) — 天井高{ceil:.0f}cm")
        print("家具だけ / 2人で運ぶ。20cm刻み、掃引のみ\n")
        print("| 階段の幅 | " + " | ".join(TYPE_LABELS[t] for t in STAIR_TYPES) + " |")
        print("|---|" + "---|" * len(STAIR_TYPES))
        for w in WIDTHS:
            cells = [f"{max_box_length(res, t, w, ceil, 'solo')} / "
                     f"{max_box_length(res, t, w, ceil, 'carried')}" for t in STAIR_TYPES]
            print(f"| {w:.0f} | " + " | ".join(cells) + " |")

    names = [(n, l, w, h) for n, l, w, h in furniture_list() if not n.startswith("箱")]
    for ceil in CEILS:
        for t in STAIR_TYPES:
            print(f"\n### 定番家具の余裕(cm) — {TYPE_LABELS[t]}・天井高{ceil:.0f}cm")
            print("家具だけ / 2人で運ぶ。負は不足\n")
            print("| 家具 | " + " | ".join(f"幅{w:.0f}" for w in WIDTHS) + " |")
            print("|---|" + "---|" * len(WIDTHS))
            for n, l, fw, fh in names:
                cells = []
                for w in WIDTHS:
                    r = res.get(case_key({"type": t, "width": w, "ceil": ceil,
                                          "L": l, "W": fw, "H": fh}))
                    cells.append("-" if r is None else
                                 f"{_fmt(r['solo'])} / {_fmt(r['carried'])}")
                print(f"| {n} {l:.0f}x{fw:.0f}x{fh:.0f} | " + " | ".join(cells) + " |")


def _fmt(v):
    return "不能" if v is None else f"{v:+.0f}"


def consistency_violations(res, tol=2.0):
    """正解データなしで見つけられる自己矛盾を列挙する。

    tol は掃引の刻みによる揺れ(約2cm)を見込んだ許容。
      (1) 人を付けたほうが余裕が大きい
      (2) 同じ階段・同じ断面で、長い箱のほうが余裕が大きい
      (3) 同じ家具で、広い階段のほうが余裕が小さい
      (4) 同じ家具で、天井が高いほうが余裕が小さい
    """
    out = []
    rs = list(res.values())
    for r in rs:
        if r["solo"] is not None and r["carried"] is not None and \
                r["carried"] > r["solo"] + tol:
            out.append(("人を付けたほうが余裕が大きい", r["key"],
                        f"家具だけ{r['solo']:+.1f} < 2人{r['carried']:+.1f}"))

    def group(keyfn):
        g = {}
        for r in rs:
            g.setdefault(keyfn(r), []).append(r)
        return g

    furn = lambda r: (r["L"], r["W"], r["H"])
    checks = [
        ("長い箱のほうが余裕が大きい",
         lambda r: (r["type"], r["width"], r["ceil"], r["W"], r["H"]) if r["name"].startswith("箱") else None,
         lambda r: r["L"], -1),
        ("広い階段のほうが余裕が小さい",
         lambda r: (r["type"], r["ceil"]) + furn(r), lambda r: r["width"], +1),
        ("天井が高いほうが余裕が小さい",
         lambda r: (r["type"], r["width"]) + furn(r), lambda r: r["ceil"], +1),
    ]
    for label, keyfn, order, sign in checks:
        for k, items in group(keyfn).items():
            if k is None:
                continue
            items = sorted(items, key=order)
            for tag in ("solo", "carried"):
                for a, b in zip(items, items[1:]):
                    if a[tag] is None or b[tag] is None:
                        continue
                    # sign=+1: 並び順に余裕が増えるはず / -1: 減るはず
                    if sign * (b[tag] - a[tag]) < -tol:
                        out.append((label, f"{a['key']} -> {b['key']}",
                                    f"{tag}: {a[tag]:+.1f} -> {b[tag]:+.1f}"))
    return out


def print_violations(res):
    v = consistency_violations(res)
    print(f"\n### 自己矛盾チェック: {len(v)} 件")
    by = {}
    for label, key, detail in v:
        by.setdefault(label, []).append((key, detail))
    for label, items in by.items():
        print(f"\n- {label}: {len(items)} 件")
        for key, detail in items[:15]:
            print(f"    {key}  {detail}")
        if len(items) > 15:
            print(f"    ...ほか {len(items) - 15} 件")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--limit", type=int, help="未計算の先頭N件だけ回す(動作確認用)")
    ap.add_argument("--table", action="store_true", help="保存済みの結果から表だけ出す")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    done = load_results(args.out)
    if not args.table:
        todo = [c for c in all_cases() if case_key(c) not in done]
        if args.limit:
            todo = todo[:args.limit]
        print(f"全{len(all_cases())}件 / 済{len(done)}件 / 今回{len(todo)}件 / 並列{args.workers}",
              flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex, \
                open(args.out, "a", encoding="utf-8") as f:
            futs = {ex.submit(run_case, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # 1件の失敗で全体を止めない。失敗は保存せず再実行に回す
                    print(f"  失敗 {case_key(c)}: {type(e).__name__}: {e}", flush=True)
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                done[r["key"]] = r
                if i % 10 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  経過{el/60:.0f}分  残り約{el/i*(len(todo)-i)/60:.0f}分",
                          flush=True)
    print_tables(done)
    print_violations(done)
