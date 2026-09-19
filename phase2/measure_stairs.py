"""
スキャンした階段から寸法(蹴上げ・踏み面・幅・段数)を計測する。

スキャンは「見た目」ではなく「数値の計測」に使う、という方針の実装。
ここで得た寸法を virtual_demo / phase2_lstair.configure に渡せば、
きれいな仮想階段を実データ準拠の寸法で組める。

信頼度の目安(この単一スキャンでの実測):
  - 蹴上げ(rise)  : 水平面の高さピーク間隔から。比較的安定
  - 段数/全高      : 全高÷蹴上げ。安定
  - 踏み面(tread) : 中心線の勾配から。ノイズに弱く過小に出やすい(要確認)
  - 幅(width)     : 各高さ帯の水平の狭い側。壁の写り込みで過大に出やすい(要確認)
"""

import numpy as np

import scan_demo as sd


def measure(path, assume_units=None):
    """スキャンOBJ/PLYから階段寸法を計測して dict で返す。

    水平な面(踏み面・踊り場・床)だけを使うのが要点。壁など垂直面を
    除外することで、幅が壁を巻き込んで過大になるのを防ぎ、段レベルごとの
    重心の水平移動から踏み面を求める(中心線の勾配より安定・正確)。

    戻り値キー: rise, tread, width, n_steps_total, total_height, confidence
    """
    m, _ = sd.load_scan_zup(path, assume_units)
    v = np.asarray(m.vertices)
    f = np.asarray(m.faces)
    n = np.asarray(m.face_normals)
    total_height = float(np.ptp(v[:, 2]))

    # 水平な面(法線がほぼ鉛直)の中心 = 踏み面・踊り場・床の点
    horiz = np.abs(n[:, 2]) > 0.9
    fc = v[f[horiz]].mean(axis=1)
    z = fc[:, 2]

    # --- 蹴上げ: 水平面の高さヒストのピーク間隔 ---
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 2, 2))
    peaks = [edges[i] + 1 for i in range(1, len(hist) - 1)
             if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]
             and hist[i] > hist.max() * 0.12]
    gaps = np.diff(sorted(peaks)) if len(peaks) >= 2 else np.array([])
    gaps = gaps[(gaps > 8) & (gaps < 30)]
    rise = float(np.median(gaps)) if len(gaps) else 18.0

    # --- 段レベルに割り当て(蹴上げ間隔)、各レベルの踏み面重心 ---
    lo = z.min()
    levels = {}
    for pz, pt in zip(z, fc):
        levels.setdefault(int(round((pz - lo) / rise)), []).append(pt)
    cent = []
    for k in sorted(levels):
        pts = np.array(levels[k])
        if len(pts) >= 8:
            cent.append((k, float(np.median(pts[:, 0])), float(np.median(pts[:, 1])), pts))
    n_steps_total = len(cent)

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
    width = float(np.median(widths)) if widths else 90.0

    # --- 天井(頭上クリアランス): 中心線の各柱で、歩行面と真上の構造の隙間 ---
    line = sd.centerline(m)
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
        "confidence": {
            "rise": "安定(高さピーク間隔)",
            "n_steps/total_height": "安定",
            "tread": "改善(段レベルの水平移動の中央値)",
            "width": "改善(踏み面のみ・壁を除外)",
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
    print(f"  段数(全体)       = {r['n_steps_total']} (2フライトなら各 {r['n_steps_total']//2})")
    print("注意: 踏み面・幅は水平面(踏み面)のみから計測して改善済みだが、"
          "スキャンのノイズで数cmの誤差は残る。virtual_demoで--tread/--width上書き可。")
