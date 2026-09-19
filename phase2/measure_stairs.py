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

    戻り値キー: rise, tread, width, n_steps_total, total_height,
                confidence(各項目の信頼度メモ)
    """
    m, _ = sd.load_scan_zup(path, assume_units)
    v = np.asarray(m.vertices)
    f = np.asarray(m.faces)
    n = np.asarray(m.face_normals)
    z = v[:, 2]
    total_height = float(np.ptp(z))

    # --- 蹴上げ: 水平な面(法線ほぼ鉛直)の中心高さのピーク間隔 ---
    horiz = np.abs(n[:, 2]) > 0.85
    fz = v[f[horiz]].mean(axis=1)[:, 2]
    hist, edges = np.histogram(fz, bins=np.arange(fz.min(), fz.max() + 2, 2))
    peaks = [edges[i] + 1 for i in range(1, len(hist) - 1)
             if hist[i] > hist[i - 1] and hist[i] >= hist[i + 1]
             and hist[i] > hist.max() * 0.15]
    gaps = np.diff(sorted(peaks)) if len(peaks) >= 2 else np.array([])
    gaps = gaps[(gaps > 8) & (gaps < 30)]           # 蹴上げらしい間隔だけ
    rise = float(np.median(gaps)) if len(gaps) else 18.0
    n_steps_total = int(round(total_height / rise)) if rise > 0 else 0

    # --- 踏み面: 中心線の勾配 dz/d水平 から (tread = rise / 勾配) ---
    line = sd.centerline(m)
    d = np.diff(line, axis=0)
    dz, dh = d[:, 2], np.hypot(d[:, 0], d[:, 1])
    asc = (dz > 1) & (dh > 1)
    slope = float(np.median(dz[asc] / dh[asc])) if asc.any() else 1.0
    tread = float(rise / slope) if slope > 0 else 26.0

    # --- 幅: 各高さ帯の水平の狭い側(壁込みなので過大寄り) ---
    ws = []
    for zc in np.arange(z.min() + 40, z.max() - 40, 30):
        b = v[(z >= zc - 15) & (z < zc + 15)]
        if len(b) < 80:
            continue
        ws.append(min(np.percentile(b[:, 0], 85) - np.percentile(b[:, 0], 15),
                      np.percentile(b[:, 1], 85) - np.percentile(b[:, 1], 15)))
    width = float(np.median(ws)) if ws else 90.0

    return {
        "rise": round(rise, 1), "tread": round(tread, 1), "width": round(width, 1),
        "n_steps_total": n_steps_total, "total_height": round(total_height, 1),
        "confidence": {
            "rise": "安定(高さピーク間隔)",
            "n_steps/total_height": "安定",
            "tread": "粗い(中心線勾配・過小寄り。要確認)",
            "width": "粗い(壁の写り込みで過大寄り。要確認)",
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
    print(f"  総高 total       = {r['total_height']} cm")
    print(f"  段数(全体)       = {r['n_steps_total']} (2フライトなら各 {r['n_steps_total']//2})")
    print("注意: tread/width は単一スキャンでは粗い。virtual_demoで個別に"
          "上書き(--tread/--width)して補正できる。")
