"""
スキャンした階段から寸法(蹴上げ・踏み面・幅・段数)を計測する。

スキャンは「見た目」ではなく「数値の計測」に使う、という方針の実装。
ここで得た寸法を stair_report / phase2_lstair.configure に渡せば、
きれいな仮想階段を実データ準拠の寸法で組める。

信頼度の目安(この単一スキャンでの実測):
  - 蹴上げ(rise)  : 水平面の高さピーク間隔から。比較的安定
  - 段数/全高      : 全高÷蹴上げ。安定
  - 踏み面(tread) : 中心線の勾配から。ノイズに弱く過小に出やすい(要確認)
  - 幅(width)     : 各高さ帯の水平の狭い側。壁の写り込みで過大に出やすい(要確認)
"""

import numpy as np

import scan_ingest as si


def centerline(mesh, band=30.0, step=25.0, smooth=3):
    """高さ帯ごとの水平重心をつないで、階段の中心線(下から上への経路)を
    近似する。戻り値: (N,3)の点列(cm, Z昇順)。

    スキャンはノイズや飛び点があるので、移動平均で軽く滑らかにする。
    """
    v = np.asarray(mesh.vertices)
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    pts = []
    for z in np.arange(zmin + band, zmax - band, step):
        b = v[(v[:, 2] >= z - band / 2) & (v[:, 2] < z + band / 2)]
        if len(b) < 50:
            continue
        pts.append([b[:, 0].mean(), b[:, 1].mean(), z])
    pts = np.array(pts)
    if smooth > 1 and len(pts) >= smooth:
        k = np.ones(smooth) / smooth
        for c in (0, 1):  # x, y だけ平滑化(高さzはそのまま)
            pts[:, c] = np.convolve(pts[:, c], k, mode="same")
    return pts


def measure(path, assume_units=None):
    """スキャンOBJ/PLYから階段寸法を計測して dict で返す。

    水平な面(踏み面・踊り場・床)だけを使うのが要点。壁など垂直面を
    除外することで、幅が壁を巻き込んで過大になるのを防ぎ、段レベルごとの
    重心の水平移動から踏み面を求める(中心線の勾配より安定・正確)。

    戻り値キー: rise, tread, width, n_steps_total(蹴上げの数), total_height
    (下の階の床から上の階の床まで), widths_by_level(段ごとの幅), confidence
    """
    m, _ = si.load_scan_zup(path, assume_units)
    v = np.asarray(m.vertices)
    f = np.asarray(m.faces)
    n = np.asarray(m.face_normals)

    # 上向きの水平な面の中心 = 踏み面・踊り場・床の点。天井(下向き)は使わない。
    # 以前は上向き・下向きの両方を使っていて、天井まで段として数えていた
    horiz = n[:, 2] > 0.9
    fc = v[f[horiz]].mean(axis=1)
    fa = np.asarray(m.area_faces)[horiz]
    z = fc[:, 2]

    # --- 蹴上げ: 水平面の高さヒストのピーク間隔 ---
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 2, 2))
    peaks = [edges[i] + 1 for i in range(1, len(hist) - 1)
             if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]
             and hist[i] > hist.max() * 0.12]
    gaps = np.diff(sorted(peaks)) if len(peaks) >= 2 else np.array([])
    gaps = gaps[(gaps > 8) & (gaps < 30)]
    rise = float(np.median(gaps)) if len(gaps) else 18.0

    # --- 面積の多い高さ(山)を段・床の高さとして拾う ---
    # 蹴上げの推定値で丸めると、数mmのずれが段数ぶん積もって1段ずれる
    # (蹴上げ20.8cmを20cmとみなすと13段が14段になる)ので、1cm刻みの面積の
    # 分布から山を直接拾う。前後3cmを足した面積が大きい順に、すでに選んだ山
    # から7cm以内のものを除きながら選ぶ(10cmにすると、高さ15cmにある別の
    # 部屋の床と、高さ25cmの1段目が1つの山にまとまってしまう)
    zmin = float(z.min())
    bins = np.floor(z - zmin).astype(int)
    hist1 = np.bincount(bins, weights=fa)
    win = np.convolve(hist1, np.ones(7), mode="same")
    picked = []
    for b in np.argsort(-win):
        if win[b] < 300.0:
            break
        if all(abs(b - q) > 7 for q in picked):
            picked.append(int(b))
    picked.sort()
    groups = [np.flatnonzero(np.abs(bins - b) <= 3) for b in picked]
    g_area = np.array([fa[g].sum() for g in groups])
    g_z = np.array([float(np.average(z[g], weights=fa[g])) for g in groups])

    # --- 階段の範囲: 下の階の床から、次の広い床(上の階)まで ---
    # 段1枚の面積(中央値)よりずっと広いかたまりを「床」とみなす。上の階の床
    # より上にある面(家具の上など)は段として数えない
    big = max(5000.0, 4.0 * float(np.median(g_area)))
    floors = [k for k in range(len(groups)) if g_area[k] > big]
    k_bottom = floors[0] if floors else 0
    upper = [k for k in floors if g_z[k] - g_z[k_bottom] >= 4 * 8.0]
    k_top = upper[0] if upper else len(groups) - 1

    # --- 上の階の床から下へ、隣り合う面だけをたどって段を選ぶ ---
    # 同じ高さ帯に、階段とは別の場所の面(一段高い別の部屋の床など)が入ることが
    # ある。段は水平に隣り合って並ぶので、直前に選んだ面と隣り合わない面は除く
    from scipy.spatial import cKDTree

    def adjacent(ga, gb, reach=40.0, frac=0.3):
        a, b = fc[ga][:, :2], fc[gb][:, :2]
        near_a = (cKDTree(b).query(a)[0] < reach).mean()
        near_b = (cKDTree(a).query(b)[0] < reach).mean()
        return max(near_a, near_b) >= frac

    chain = [k_top]
    for k in range(k_top - 1, k_bottom, -1):
        if adjacent(groups[k], groups[chain[-1]]):
            chain.append(k)
    chain.append(k_bottom)
    chain = chain[::-1]

    # --- 段の間隔がそろわない面を除く ---
    # 段の上に置かれた物などが、段と段の間の高さに小さな山を作ることがある。
    # 前後どちらかの間隔が、ふつうの間隔(中央値)の6割より狭い面のうち、面積が
    # 一番小さいものから除いていく
    while len(chain) > 3:
        gz = g_z[chain]
        gap = np.diff(gz)
        med = float(np.median(gap))
        odd = [i for i in range(1, len(chain) - 1)
               if gap[i - 1] < 0.6 * med or gap[i] < 0.6 * med]
        if not odd:
            break
        chain.pop(min(odd, key=lambda i: g_area[chain[i]]))
    total_height = float(g_z[k_top] - g_z[k_bottom])
    n_steps_total = len(chain) - 1                           # 蹴上げの数
    if n_steps_total > 0:
        rise = total_height / n_steps_total

    cent = []
    for k in chain:
        pts = fc[groups[k]]
        if len(pts) >= 8:
            cent.append((k, float(np.median(pts[:, 0])), float(np.median(pts[:, 1])), pts))

    # --- 踏み面: 隣接段レベルの重心の水平移動 ---
    treads = []
    for a, b in zip(cent[:-1], cent[1:]):
        if b[0] - a[0] != 1:
            continue
        d = float(np.hypot(b[1] - a[1], b[2] - a[2]))
        if 10 < d < 50:
            treads.append(d)
    tread = float(np.median(treads)) if treads else 26.0

    # --- 幅: 各段の踏み面を、局所進行方向に直交する向きへ射影した広がり ---
    widths = []
    widths_by_level = []  # (下の階の床から数えた段の番号, 幅cm)
    for k, (_, cx, cy, pts) in enumerate(cent):
        if k + 1 < len(cent):
            d = np.array([cent[k + 1][1] - cx, cent[k + 1][2] - cy])
        elif k > 0:
            d = np.array([cx - cent[k - 1][1], cy - cent[k - 1][2]])
        else:
            continue
        nrm = np.linalg.norm(d)
        if nrm < 1e-6:
            continue
        perp = np.array([-d[1], d[0]]) / nrm
        proj = (pts[:, :2] - [cx, cy]) @ perp
        w = float(np.percentile(proj, 95) - np.percentile(proj, 5))
        if 40 < w < 160:
            widths.append(w)
            widths_by_level.append((chain.index(cent[k][0]), round(w, 1)))
    width = float(np.median(widths)) if widths else 90.0

    # --- 天井(頭上クリアランス): 中心線の各柱で、歩行面と真上の構造の隙間 ---
    line = centerline(m)
    gaps = []
    for x, y, zc in line:
        col = v[np.hypot(v[:, 0] - x, v[:, 1] - y) < 25]
        if len(col) < 20:
            continue
        below = col[col[:, 2] <= zc + 10]
        above = col[col[:, 2] > zc + 30]
        if len(below) < 5 or len(above) < 5:
            continue
        gap = np.percentile(above[:, 2], 20) - np.percentile(below[:, 2], 20)
        if 120 < gap < 260:
            gaps.append(gap)
    headroom = float(np.median(gaps)) if gaps else 220.0

    return {
        "rise": round(rise, 1), "tread": round(tread, 1), "width": round(width, 1),
        "n_steps_total": n_steps_total, "total_height": round(total_height, 1),
        "headroom": round(headroom, 1),
        "widths_by_level": widths_by_level,
        "confidence": {
            "rise": "安定(床から床までの高さ÷段数)",
            "n_steps/total_height": "安定(上向きの面の高さの山を、下の階の床から上の階の床まで数える)",
            "tread": "改善(段レベルの水平移動の中央値)",
            "width": "改善(踏み面のみ・壁を除外)。段ごとの値は扇形の段で粗い",
            "headroom": "粗い(頭上の構造の有無に依存)",
        },
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="階段のスキャン OBJ/PLY")
    ap.add_argument("--assume-units", choices=("m", "cm", "mm"))
    args = ap.parse_args()
    r = measure(args.scan, args.assume_units)
    print("スキャンからの階段寸法の計測結果:")
    print(f"  蹴上げ rise      = {r['rise']} cm   [{r['confidence']['rise']}]")
    print(f"  踏み面 tread     = {r['tread']} cm   [{r['confidence']['tread']}]")
    print(f"  幅 width         = {r['width']} cm   [{r['confidence']['width']}]")
    print(f"  頭上 headroom    = {r['headroom']} cm   [{r['confidence']['headroom']}]")
    print(f"  総高 total       = {r['total_height']} cm")
    print(f"  段数(蹴上げの数) = {r['n_steps_total']} (下の階の床から上の階の床まで)")
    if r["widths_by_level"]:
        print("  段ごとの幅(下から何段目: cm) = "
              + ", ".join(f"{k}: {w:.0f}" for k, w in r["widths_by_level"]))
    print("注意: 踏み面・幅は水平面(踏み面)のみから計測して改善済みだが、"
          "スキャンのノイズで数cmの誤差は残る。stair_reportで--tread/--width上書き可。")
