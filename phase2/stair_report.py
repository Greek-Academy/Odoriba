"""
「あなたの階段への答え」レポート: スキャンから計測した寸法で仮想階段を
組み、アプリ本来の出力(ボトルネック・上限サイズ・家具カタログ判定)を出す。

    python stair_report.py --measure-stair stair.obj
    python stair_report.py --width 90 --rise 18 --tread 26 --steps 8   # 数値直接
    python stair_report.py --measure-stair stair.obj --catalog          # 家具一覧表も
    python stair_report.py ... --quick                                  # 掃引のみ(RRTなし・速い)
"""

import argparse
import time

import numpy as np

import phase2_lstair as L
import measure_stairs as ms


def _bottleneck(outer, obstacles, with_human, num_carriers):
    sweep = L.sweep_capacity(outer, obstacles, num_carriers=num_carriers,
                             with_human=with_human)
    finite = [(s, c) for s, c, _ in sweep if np.isfinite(c)]
    s_min, c_min = min(finite, key=lambda r: r[1])
    s_land0, s_land1 = L.FL1_Y1, L.S_CORNER + (L.FL2_X0 - L.CENTER)
    where = "踊り場" if s_land0 <= s_min <= s_land1 else f"弧長{s_min:.0f}cm地点"
    return where, c_min


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure-stair", metavar="SCAN.obj")
    ap.add_argument("--width", type=float)
    ap.add_argument("--rise", type=float)
    ap.add_argument("--tread", type=float)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--catalog", action="store_true", help="定番家具の一覧表も出す(遅い)")
    ap.add_argument("--quick", action="store_true", help="掃引のみ(RRTを回さず速い)")
    ap.add_argument("--max-iter", type=int, default=4000)
    args = ap.parse_args()

    width, rise, tread = args.width, args.rise, args.tread
    n = args.steps
    if args.measure_stair:
        r = ms.measure(args.measure_stair)
        print(f"スキャン計測: 蹴上げ{r['rise']:.0f} / 踏み面{r['tread']:.0f} / "
              f"幅{r['width']:.0f} / 全高{r['total_height']:.0f} / 段数{r['n_steps_total']}")
        width = width if width is not None else r["width"]
        rise = rise if rise is not None else r["rise"]
        tread = tread if tread is not None else r["tread"]
        n = n if n is not None else max(1, r["n_steps_total"] // 2)
    L.configure(width=width, rise=rise, tread=tread, n_steps1=n, n_steps2=n)
    outer, obstacles = L.build_lstairs()
    print(f"\n=== あなたの階段: 幅{L.STAIR_WIDTH:.0f}cm / 蹴上げ{L.RISE:.0f} x "
          f"踏み面{L.TREAD:.0f}cm / ({L.N_STEPS1}+{L.N_STEPS2})段 / "
          f"踊り場{L.STAIR_WIDTH:.0f}角 / 全高{L.TOP_Z:.0f}cm ===")

    # --- ボトルネック(掃引・速い): 標準家具でどこが一番狭いか ---
    print(f"\n[ボトルネック] 基準家具 {L.FURN_L:.0f}x{L.FURN_W:.0f}x{L.FURN_H:.0f}cm")
    t0 = time.time()
    where_b, c_b = _bottleneck(outer, obstacles, with_human=False, num_carriers=None)
    where_h, c_h = _bottleneck(outer, obstacles, with_human=True, num_carriers=2)
    print(f"  家具単体: 最も狭いのは{where_b}、最良姿勢での余裕 {c_b:.0f}cm")
    if c_h >= 0:
        print(f"  2人運搬: 最も狭いのは{where_h}、余裕 {c_h:.0f}cm")
    else:
        print(f"  2人運搬: {where_h}で あと{-c_h:.0f}cm 足りない")
    print(f"  [{time.time()-t0:.0f}s]")

    if args.quick:
        raise SystemExit(0)

    # --- 上限サイズ(RRT二分探索) ---
    print("\n[上限サイズ] 二分探索中(RRTを繰り返すので数分)...")
    t0 = time.time()
    max_w, _ = L.find_max_furniture_width(outer, obstacles, max_iter=args.max_iter)
    print(f"  家具単体なら 幅{max_w:.0f}cm まで入る [{time.time()-t0:.0f}s]")
    t0 = time.time()
    max_l, _ = L.find_max_furniture_length(outer, obstacles, num_carriers=2,
                                           max_iter=args.max_iter)
    if max_l is None:
        print(f"  2人運搬: 試した長さでは通らなかった [{time.time()-t0:.0f}s]")
    else:
        print(f"  2人で運ぶなら 長さ{max_l:.0f}cm まで入る [{time.time()-t0:.0f}s]")

    if args.catalog:
        import catalog
        print("\n[定番家具カタログ] あなたの階段で通るか(遅い)...")
        results = []
        for name, l, w, h, note in catalog.CATALOG:
            res = catalog.judge_one(l, w, h, outer, obstacles, max_iter=args.max_iter)
            res.update(name=name, L=l, W=w, H=h, note=note)
            results.append(res)
            print(f"  {name}: 単体={catalog._verdict(res['box'])} / "
                  f"2人={catalog._verdict(res['carriers'])}")
