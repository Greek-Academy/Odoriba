"""
スキャンから、階ごとの間取り図(上から見た図)を作る。

scan_picker の「出発点と目的地を床に置く」図に使う。以前は撮れた床の点を
高さで色分けして出していたが、壁も部屋の形も無く、階段のように床が上下に
重なる所は利用者が高さを数字で絞る必要があった。ここでは階を自動で分け、
階ごとに「床・階段・家具など・壁」のマス目の図にする。

判定(オラクル・掃引)には使わない。図を描くためだけの、粗い分類。

メッシュは Z が上・cm(scan_ingest.load_scan_zup の戻り値)。
"""

import numpy as np

# 階を見つけるときの値。実データ(家全体・階段だけのスキャン)で、床の面積を
# 高さ10cmごとに集計すると、階は数㎡の山になり、段は1段あたり0.15〜0.3㎡、
# 踊り場は1㎡弱だった。
LEVEL_BIN = 10.0          # 高さを集計する刻み(cm)
LEVEL_MIN_AREA = 1.0e4    # これ以上の面積(cm²)がある高さを階の候補にする
LEVEL_MIN_RATIO = 0.2     # 一番広い高さに対して、これ未満の山は階にしない
LEVEL_MERGE = 100.0       # この高さの差(cm)未満の山は同じ階(玄関の段差など)


def find_levels(z, area):
    """上向きの面の高さ z と面積 area から、階の床の高さ(低い順)を返す。

    面積の大きい高さを階の候補にし、近い候補は1つの階にまとめる。まとめた
    中で一番広い高さを、その階の床の高さにする(机やベッドの上面が床より
    狭ければ、床のほうが選ばれる)。小さなスキャンで候補が1つも無いときは、
    一番広い高さを階にする。
    """
    z = np.asarray(z, dtype=float)
    area = np.asarray(area, dtype=float)
    if len(z) == 0:
        return []
    lo = np.floor(z.min() / LEVEL_BIN) * LEVEL_BIN
    idx = ((z - lo) // LEVEL_BIN).astype(int)
    hist = np.bincount(idx, weights=area)
    need = max(LEVEL_MIN_AREA, LEVEL_MIN_RATIO * hist.max())
    big = np.flatnonzero(hist >= need)
    if len(big) == 0:
        big = np.array([int(np.argmax(hist))])
    groups = [[int(big[0])]]
    for b in big[1:]:
        if (b - groups[-1][-1]) * LEVEL_BIN < LEVEL_MERGE:
            groups[-1].append(int(b))
        else:
            groups.append([int(b)])
    levels = []
    for g in groups:
        b = max(g, key=lambda i: hist[i])
        m = idx == b
        levels.append(float(np.average(z[m], weights=area[m])))
    return levels
