"""
家具店・家電量販店が公開している搬入の目安を式にして、寸法の掃引
(param_sweep.py)の判定と突き合わせる。

    python store_rules.py            # 保存済みの results/param_sweep.jsonl と突き合わせる
    python store_rules.py --points   # 公開されている実例(通った/断られた)をオラクルで判定する

目安はどれも「家具を縦に立てて運ぶ」前提の辺の比較で、運ぶ人の体は
余裕(+10〜20cm)に丸められている。出典と確からしさは docs/decisions.md
(2026-11-15)。家具の3辺を長い順に A >= B >= C とする。
"""

import argparse
import math

import param_sweep as ps


def _abc(l, w, h):
    return sorted((l, w, h), reverse=True)


def rule_receno(width, ceil, l, w, h):
    """Re:CENO: 階段の幅がC以上・曲がり角の対角線がB以上・高さがA以上。
    掃引の環境は踊り場が幅と同じ一辺の正方形なので、対角線は √2・幅。"""
    a, b, c = _abc(l, w, h)
    return c <= width and b <= math.sqrt(2) * width and a <= ceil


def rule_plus10(width, ceil, l, w, h):
    """メーカー各社・ビックカメラ: 本体の幅と高さに+10cm(シャープは幅・高さとも)。"""
    a, b, c = _abc(l, w, h)
    return c + 10 <= width and a + 10 <= ceil


def rule_plus20(width, ceil, l, w, h):
    """ケーズ・ヨドバシ・ノジマ(二次情報): 階段は狭い方の辺+20cm。"""
    a, b, c = _abc(l, w, h)
    return c + 20 <= width and a <= ceil


def rule_joshin_turn(width, ceil, l, w, h):
    """Joshin(検索結果の要約のみ・未確認): 曲がる階段は広い方の辺+20cm。
    縦に立てたときの広い方の辺 = B。"""
    a, b, c = _abc(l, w, h)
    return b + 20 <= width and a <= ceil


RULES = [
    ("Re:CENO(辺の比較)", rule_receno),
    ("+10cm(メーカー各社)", rule_plus10),
    ("+20cm(量販店の階段)", rule_plus20),
    ("曲がりは広い辺+20cm(Joshin)", rule_joshin_turn),
]

# 公開されている実例。(説明, 型, 幅, 天井高, (L, W, H), 現実の結果)
# 幅は手すりの内側。型が書かれていないものはL字で代用している
TEST_POINTS = [
    ("価格.com: 75cm幅(手すり内67)・90度曲がり・最低高177cm。店の言う上限 175x65x67",
     "lstair", 67.0, 177.0, (175.0, 67.0, 65.0), True),
    ("同上。179.8x60x66.9 の冷蔵庫は断られた",
     "lstair", 67.0, 177.0, (179.8, 66.9, 60.0), False),
    ("Re:CENO: 梱包180x92x68のソファ、手すり内82cmの階段で搬入可",
     "lstair", 82.0, 220.0, (180.0, 92.0, 68.0), True),
    ("Panasonic: ドラム式(約64x72x105)は幅71cmあればほぼ搬入可",
     "lstair", 71.0, 220.0, (105.0, 72.0, 64.0), True),
]


def compare(res):
    """掃引の各ケースで、目安の可否とオラクルの可否(家具だけ/2人)を数える。"""
    rs = [r for r in res.values() if r["solo"] is not None and r["carried"] is not None]
    print(f"\n### 目安とオラクルの一致(全{len(rs)}件)")
    print("各セル: 目安○オラクル○ / 目安○オラクル× / 目安×オラクル○ / 目安×オラクル×\n")
    print("| 目安 | 目安が○の件数 | 家具だけ | 2人で運ぶ |")
    print("|---|---|---|---|")
    for label, fn in RULES:
        cells = []
        n_ok = 0
        for tag in ("solo", "carried"):
            k = {(a, b): 0 for a in (True, False) for b in (True, False)}
            for r in rs:
                k[(fn(r["width"], r["ceil"], r["L"], r["W"], r["H"]), r[tag] >= 0)] += 1
            n_ok = k[(True, True)] + k[(True, False)]
            agree = (k[(True, True)] + k[(False, False)]) / len(rs) * 100
            cells.append(f"{k[(True, True)]} / {k[(True, False)]} / {k[(False, True)]} / "
                         f"{k[(False, False)]} (一致{agree:.0f}%)")
        print(f"| {label} | {n_ok} | " + " | ".join(cells) + " |")

    # 目安が○なのに2人で通らないケースを、不足の大きさで分ける
    print("\n### Re:CENO の目安では○なのに、2人では通らないケースの不足量")
    short = [-r["carried"] for r in rs
             if rule_receno(r["width"], r["ceil"], r["L"], r["W"], r["H"]) and r["carried"] < 0]
    for lo, hi in ((0, 5), (5, 15), (15, 30), (30, 1e9)):
        n = sum(lo < s <= hi for s in short)
        print(f"  不足 {lo}〜{'∞' if hi > 1e8 else hi}cm: {n} 件")


def run_points():
    print("\n### 公開されている実例をオラクルで判定")
    print("| 実例 | 現実 | 家具だけ | 2人で運ぶ |")
    print("|---|---|---|---|")
    for desc, t, width, ceil, (l, w, h), real in TEST_POINTS:
        r = ps.run_case({"type": t, "width": width, "ceil": ceil, "name": desc,
                         "L": l, "W": w, "H": h})
        print(f"| {desc} | {'通った' if real else '断られた'} | "
              f"{ps._fmt(r['solo'])}cm | {ps._fmt(r['carried'])}cm |", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", action="store_true", help="公開されている実例を判定する(数分)")
    args = ap.parse_args()
    if args.points:
        run_points()
    else:
        compare(ps.load_results())
