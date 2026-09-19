"""
実スキャン階段のデモ: iPhone(Scaniverse)でスキャンした階段のOBJを
取り込み、その形状の上に家具を置いてクリアランスを計算・可視化する。

パイプライン: scan_ingest で単位をcmに正規化 -> Z-up に軸合わせ ->
MeshEnv(応急モード=符号なし距離)で家具のクリアランスを評価 ->
上面図・側面図に scan の点群と家具を重ねて描く。

いまの段階の割り切り(実データ一次デモ):
  - スキャンは開いた面なので水密化せず、MeshEnvの符号なし距離
    (家具の表面点から一番近いスキャン面までの距離)で「壁・段への
    近さ」を測る。値が大きい=周囲の面から離れている。0付近=面に接触/貫入。
    内外の厳密判定(めり込みの符号)や経路探索(RRT)は、開口部に蓋をして
    水密化してから(次段階)。
  - 目的はまず「実際にスキャンした階段の中に家具を置いて、数字と絵が
    出る」ことの実証。

実行:
    python scan_demo.py <scan.obj>
    python scan_demo.py <scan.obj> --pos X Y Z --yaw 90 --pitch 30
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh

import phase2_lstair as L
import mesh_env as me
import scan_ingest as si


def load_scan_zup(path, assume_units=None):
    """スキャンOBJを読み、cmに正規化し、Z-upに軸合わせして原点を揃える。

    Scaniverseは重力方向がY軸。本プロジェクトは高さがZ軸なので、
    Y->Z, Z->-Y の入れ替え(X軸まわり-90度)で立てる。
    戻り値: (mesh_cm_zup, info)
    """
    m = si.load_mesh(path)
    scale, unit_label = si.guess_units_scale(m, assume_units)
    m.apply_scale(scale)
    # Y-up -> Z-up
    R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    m.apply_transform(R)
    # 原点を最小コーナーに揃える(座標を正の領域に置く)
    m.apply_translation(-m.bounds[0])
    info = {"unit_label": unit_label, "extents_cm": [float(e) for e in m.extents]}
    return m, info


def auto_start_pose(mesh):
    """一次デモ用の家具姿勢の当て推量: 足元付近(低いZ)の水平中心に、
    長軸を水平にして置く。実データに合わせて --pos/--yaw で上書きできる。"""
    v = np.asarray(mesh.vertices)
    # 低い方から15%の点の重心を「階段の下り口あたり」とみなす
    z = v[:, 2]
    low = v[z < np.percentile(z, 15)]
    cx, cy = low[:, 0].mean(), low[:, 1].mean()
    cz = np.percentile(z, 15) + L.FURN_H / 2 + 20.0
    quat = L._pose(0.0, 0.0).as_quat()
    return np.array([cx, cy, cz]), quat


def render(mesh, pos, quat, clearance, out_path):
    """scanの点群(灰)に家具(青)を重ねて、上面図と側面図で描く。"""
    v = np.asarray(mesh.vertices)
    fpts = L.furniture_points(pos, quat)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    views = [((0, 1), "top (X-Y)"), ((0, 2), "side (X-Z, Z=height)")]
    for a, ((i, j), t) in zip(ax, views):
        a.scatter(v[::15, i], v[::15, j], s=1, c="#b8b8b8", zorder=0)
        a.scatter(fpts[:, i], fpts[:, j], s=8, c="tab:blue", zorder=2)
        a.scatter([pos[i]], [pos[j]], s=40, c="tab:red", marker="x", zorder=3)
        a.set_title(t)
        a.set_aspect("equal")
    fig.suptitle(f"Real scanned staircase + furniture "
                 f"({L.FURN_L:.0f}x{L.FURN_W:.0f}x{L.FURN_H:.0f}cm) / "
                 f"min distance to scan surface = {clearance:.1f}cm  "
                 f"(unsigned, approx)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", help="スキャンした OBJ/PLY")
    parser.add_argument("--assume-units", choices=("m", "cm", "mm"))
    parser.add_argument("--pos", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="家具の中心位置(cm)。省略時は自動当て推量")
    parser.add_argument("--yaw", type=float, default=0.0, help="ヨー(度)")
    parser.add_argument("--pitch", type=float, default=0.0, help="ピッチ=傾き(度)")
    parser.add_argument("--out", default="scan_demo.png")
    args = parser.parse_args()

    mesh, info = load_scan_zup(args.scan, args.assume_units)
    print(f"取り込み: 単位={info['unit_label']}, "
          f"サイズ(cm)= {info['extents_cm'][0]:.0f} x {info['extents_cm'][1]:.0f} "
          f"x {info['extents_cm'][2]:.0f} (Z=高さ)")

    env = me.MeshEnv(mesh, require_watertight=False)

    if args.pos is not None:
        pos = np.array(args.pos)
        quat = L._pose(args.yaw, args.pitch).as_quat()
    else:
        pos, quat = auto_start_pose(mesh)
        print(f"家具姿勢(自動): 位置={pos.round(0)}, 水平・傾き0")

    clearance = env.furniture_clearance(pos, quat)
    print(f"家具表面からスキャン面までの最小距離: {clearance:.1f}cm "
          "(符号なし=近さの目安。0付近で面に接触/貫入)")
    render(mesh, pos, quat, clearance, args.out)
