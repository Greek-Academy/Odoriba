"""
定番家具カタログの一斉判定: よくある家具をL字階段(phase2_lstair)に
流して、「単体で通るか / 2人で運んで通るか / 余裕・不足は何cm」を
一覧表にする。

    python catalog.py                 # 本番(全RRT実行で15〜30分)
    python catalog.py --max-iter 200  # 動作確認用(判定は当てにならない)
    python catalog.py --table-only    # 保存済み results/catalog.json から表だけ再表示

出力: results/catalog.json(コミットする) と、標準出力のMarkdown表。

判定の意味(いつもの注意):
  - PASS はRRTが実際の運び方を見つけた、という構成的な証明
  - BLOCKED は「この試行回数では見つからなかった」+ ボトルネック掃引の
    不足量(負のcm)が裏付け
  - 家具は剛体と仮定する。マットレスのように曲がる物は実際より厳しい
    判定になる(表の備考に明記)
"""

import argparse
import json
import os
import time

import numpy as np

import phase2_demo as p
import phase2_lstair as L

# (名前, L=長辺, W=短辺, H=高さ, 備考)。寸法は日本の量販店でよくある
# サイズの代表値(cm)。W/Hの割り当ては「実際に運ぶときの向き」で決める
# (Hがワールド上下方向)。
CATALOG = [
    ("冷蔵庫(400Lクラス)", 180, 65, 60, "横に寝かせて運ぶ想定(実際は縦運搬が多い点は簡略化)"),
    ("ドラム式洗濯機", 105, 72, 60, ""),
    ("2人掛けソファ", 160, 85, 80, ""),
    ("3人掛けソファ", 200, 90, 85, ""),
    ("本棚(高さ200cm)", 200, 90, 30, "背板を下に寝かせて運ぶ想定"),
    ("セミダブルマットレス", 195, 25, 120, "立てて(厚み25cmを幅に)運ぶ想定。剛体仮定なので実際より厳しい"),
    ("タンス(検証シナリオ)", 200, 50, 65, "これまでの検証シナリオと同一"),
]

NUM_CARRIERS = 2


def judge_one(l, w, h, outer, obstacles, max_iter=4000, seeds=(0, 1)):
    """1つの家具の判定。phase2_lstairの寸法グローバルを一時的に
    上書きして(find_max_*と同じパターン)、単体/運搬者ありの両方を
    RRT+ボトルネック掃引で評価する。"""
    orig = (L.FURN_L, L.FURN_W, L.FURN_H)
    L.FURN_L, L.FURN_W, L.FURN_H = float(l), float(w), float(h)
    try:
        # START/GOALのzは家具の高さに依存する(床の上に置いた状態)ので、
        # モジュール定数ではなくここで組み立てる
        start = (np.array([L.CENTER, 170.0, h / 2]), L._pose(90.0, 0.0).as_quat())
        goal = (np.array([L.FL2_X1 + 170.0, L.LAND_YC, L.TOP_Z + h / 2]),
                L._pose(0.0, 0.0).as_quat())
        out = {}
        for with_human, key in ((False, "box"), (True, "carriers")):
            found = False
            t0 = time.time()
            for sd in seeds:
                path = p.rrt_connect(
                    start, goal, with_human=with_human, outer=outer, obstacles=obstacles,
                    num_carriers=NUM_CARRIERS, max_iter=max_iter, seed=sd, w_rot=L.W_ROT,
                    sampler=L.make_sampler(with_human=with_human), validator=L.state_valid)
                if path is not None:
                    found = True
                    break
            sweep = L.sweep_capacity(outer, obstacles, num_carriers=NUM_CARRIERS,
                                     with_human=with_human)
            finite = [(s, c) for s, c, _ in sweep if np.isfinite(c)]
            s_min, c_min = min(finite, key=lambda r: r[1])
            out[key] = {"found": found, "capacity_cm": float(c_min), "s": float(s_min),
                        "time_s": time.time() - t0}
        return out
    finally:
        L.FURN_L, L.FURN_W, L.FURN_H = orig


def _verdict(entry):
    """表のセル: PASS(最狭で余裕◯cm) / BLOCKED(あと◯cm)。

    RRTの結論と掃引の符号が食い違う場合(掃引は正なのに経路が見つから
    ない等)は、回転の途中姿勢が原因のことが多いので両方を出す。
    """
    c = entry["capacity_cm"]
    if entry["found"]:
        return f"PASS (最狭で余裕{c:.0f}cm)" if c >= 0 else f"PASS (余裕ほぼ0)"
    if c < 0:
        return f"BLOCKED (あと{-c:.0f}cm)"
    return f"BLOCKED (静止姿勢は置けるが回しきる経路が見つからず)"


def print_table(results):
    print("\n| 家具 | 寸法 LxWxH (cm) | 単体 | 2人で運搬 | 備考 |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['L']:.0f}x{r['W']:.0f}x{r['H']:.0f} "
              f"| {_verdict(r['box'])} | {_verdict(r['carriers'])} | {r['note']} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--out", default=os.path.join("results", "catalog.json"))
    parser.add_argument("--table-only", action="store_true",
                        help="保存済みJSONから表だけ表示する(計算しない)")
    args = parser.parse_args()

    if args.table_only:
        with open(args.out) as f:
            print_table(json.load(f)["furniture"])
        raise SystemExit(0)

    outer, obstacles = L.build_lstairs()
    print(f"L字階段(幅{L.STAIR_WIDTH:.0f}cm・踊り場{L.STAIR_WIDTH:.0f}x{L.STAIR_WIDTH:.0f}cm)に"
          f"{len(CATALOG)}種類の家具を流します(RRT 2seeds x 2ケース、数分/件)")
    results = []
    for name, l, w, h, note in CATALOG:
        t0 = time.time()
        r = judge_one(l, w, h, outer, obstacles, max_iter=args.max_iter)
        r.update(name=name, L=l, W=w, H=h, note=note)
        results.append(r)
        print(f"  {name}: 単体={_verdict(r['box'])} / 2人={_verdict(r['carriers'])} "
              f"[{time.time() - t0:.0f}s]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"stair": "phase2_lstair (width 90, landing 90x90, 6+6 steps)",
                   "num_carriers": NUM_CARRIERS, "max_iter": args.max_iter,
                   "furniture": results}, f, indent=1, ensure_ascii=False)
    print(f"\nsaved to {args.out}")
    print_table(results)
