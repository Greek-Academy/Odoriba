"""
スキャンメッシュの取り込み前処理: iPhone LiDAR で撮った階段の
OBJ/PLY を、MeshEnv(phase2/mesh_env.py)が使える「自由空間の水密
メッシュ」に整える。

生スキャンは判定にそのまま使えない。主に3つ整える:
  1. 単位  -- スキャンはメートルで書き出されることが多いが、本プロジェクト
     は cm。バウンディングボックスの大きさから推定して cm に正規化する
  2. 水密化 -- スキャンは「壁の内面」の開いた面で、穴・欠けがある。
     まず穴埋め(trimesh.repair)、それで水密にならなければボクセル化
     してから marching cubes で閉じた面に作り直す
  3. 向き  -- (最終的な軸合わせはSTART/GOALと一緒に実データを見て
     決めるので、ここでは主軸(PCA)で大まかに揃える下ごしらえだけ)

前提: スキャンは「囲まれた空間の内面」で、閉じれば内部が自由空間になる
(部屋・階段室の内側をなめるスキャンならこの前提が成り立つ)。

実行:
    python scan_ingest.py <mesh.obj>                 # 診断+水密化+書き出し
    python scan_ingest.py <mesh.obj> --assume-units m
    python scan_ingest.py --selftest                 # 実スキャンなしの自己テスト
"""

import argparse
import os

import numpy as np
import trimesh


def load_mesh(path):
    """OBJ/PLYを1つのTrimeshとして読む(Sceneなら結合する)。"""
    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"メッシュとして読めなかった: {path}")
    return m


def guess_units_scale(mesh, assume=None):
    """cmへの変換係数を返す。assume('m'|'cm'|'mm')があればそれを優先。

    推定は最大辺の長さから: 一般的な階段室は数メートル。
      最大辺 < 15   -> メートル (x100)
      15..1500      -> センチ   (x1、そのまま)
      > 1500        -> ミリ     (x0.1)
    """
    factor = {"m": 100.0, "cm": 1.0, "mm": 0.1}
    if assume:
        return factor[assume], assume + "(指定)"
    ext = float(np.max(mesh.bounding_box.extents))
    if ext < 15:
        return 100.0, "m(推定)"
    if ext > 1500:
        return 0.1, "mm(推定)"
    return 1.0, "cm(推定)"


def diagnose(mesh, label="mesh"):
    """メッシュの診断を表示して、要点をdictで返す。"""
    ext = mesh.bounding_box.extents
    info = {
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "watertight": bool(mesh.is_watertight),
        "extents": [float(e) for e in ext],
        "volume": float(mesh.volume) if mesh.is_watertight else None,
    }
    print(f"[{label}] 面={info['faces']} 頂点={info['vertices']} "
          f"水密={info['watertight']}")
    print(f"  バウンディングボックス(現単位): "
          f"{ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f}")
    return info


def make_watertight(mesh, voxel_pitch=None):
    """自由空間の水密メッシュにする。

    (1) 法線を直して穴埋めを試す。これで水密になれば軽くて正確なので採用。
    (2) だめならボクセル化 -> 内部を塗りつぶし -> marching cubes で
        閉じた面に作り直す(穴だらけのスキャンに強いが、ボクセル解像度
        なりに角が丸まる)。
    戻り値: (mesh, method)  method = "repair" | "voxel"
    """
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.unique_faces())        # 重複面を除く(trimesh 5系のAPI)
    m.update_faces(m.nondegenerate_faces())  # つぶれた面を除く
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    trimesh.repair.fill_holes(m)
    if m.is_watertight:
        return m, "repair"

    # ボクセル化して穴を無視して体積として閉じる
    if voxel_pitch is None:
        voxel_pitch = float(np.max(mesh.bounding_box.extents)) / 128.0
    vox = mesh.voxelized(pitch=voxel_pitch).fill()
    filled = vox.marching_cubes
    trimesh.repair.fix_normals(filled)

    # 体積の妥当性チェック: 出入口が開いたままのスキャン(階段室は
    # 上下が開いている)だと、内部の塗りつぶしが穴から漏れて、水密には
    # なっても中身のない薄い殻(体積ほぼ0)になる。バウンディングボックス
    # 体積に対して極端に小さければ、開口部に蓋をする前処理が要るサイン。
    bbox_vol = float(np.prod(mesh.bounding_box.extents))
    if bbox_vol > 0 and filled.volume < 0.02 * bbox_vol:
        print("警告: 水密化後の体積が極端に小さい。スキャンの開口部"
              "(階段の出入口など)が閉じておらず塗りつぶしが漏れている可能性大。"
              "開口に蓋をする前処理が必要(実データを見て対応)。")
    return filled, "voxel"


def preprocess(path, assume_units=None, voxel_pitch=None, out_path=None):
    """load -> 単位正規化(cm) -> 水密化 -> 書き出し、までを一括で行う。

    戻り値: (mesh_cm_watertight, report)
    """
    raw = load_mesh(path)
    print(f"読み込み: {path}")
    diagnose(raw, "生スキャン")

    scale, unit_label = guess_units_scale(raw, assume_units)
    if scale != 1.0:
        raw.apply_scale(scale)
    print(f"単位: {unit_label} -> cmへ x{scale}")
    ext_cm = raw.bounding_box.extents
    print(f"  cm換算のバウンディングボックス: "
          f"{ext_cm[0]:.0f} x {ext_cm[1]:.0f} x {ext_cm[2]:.0f} cm")

    water, method = make_watertight(raw)
    print(f"水密化: {method} -> 水密={water.is_watertight}")
    diagnose(water, "水密化後")

    if out_path:
        water.export(out_path)
        print(f"書き出し: {out_path} (git管理外)")

    report = {
        "unit_label": unit_label, "scale": scale,
        "extents_cm": [float(e) for e in ext_cm],
        "watertight": bool(water.is_watertight), "method": method,
    }
    return water, report


# ---- 自己テスト(実スキャンなしで前処理の各段が動くか) ----

def _selftest():
    """箱で作ったL字階段メッシュを疑似スキャンに見立てて、
    (a)単位換算 (b)穴あきメッシュの水密化 が動くことを確認する。"""
    import phase2_lstair as L
    import mesh_env as me

    print("=== 自己テスト: 箱L字階段メッシュを疑似スキャンとして通す ===")
    outer, obstacles = L.build_lstairs()
    free = me.env_to_mesh(outer, obstacles)  # cm・水密の自由空間メッシュ
    print(f"合成した自由空間メッシュ: 水密={free.is_watertight}, "
          f"体積={free.volume / 1e6:.1f} m^3")

    # (a) わざとメートル単位に落として、単位推定が cm に戻せるか
    m_scaled = free.copy()
    m_scaled.apply_scale(0.01)  # cm -> m 相当
    scale, label = guess_units_scale(m_scaled)
    print(f"(a) 単位推定: {label}, x{scale} "
          f"-> {'OK' if abs(scale - 100.0) < 1e-9 else 'NG'}")

    # (b) 面を1割ほど削って穴あきにし、水密化で復活するか
    holed = free.copy()
    keep = np.ones(len(holed.faces), dtype=bool)
    rng = np.random.default_rng(0)
    keep[rng.choice(len(holed.faces), size=len(holed.faces) // 10, replace=False)] = False
    holed.update_faces(keep)
    print(f"(b) 穴あきメッシュ: 水密={holed.is_watertight}")
    fixed, method = make_watertight(holed)
    print(f"    水密化({method}) -> 水密={fixed.is_watertight} "
          f"-> {'OK' if fixed.is_watertight else 'NG'}")

    # (c) 水密化したメッシュを MeshEnv に渡して判定が動くか
    env = me.MeshEnv(fixed, require_watertight=False)
    c = env.furniture_clearance(*L.START)
    print(f"(c) MeshEnvでSTART姿勢のクリアランス: {c:.1f}cm "
          f"-> {'OK(数値が出た)' if np.isfinite(c) else 'NG'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", nargs="?", help="スキャンした OBJ/PLY")
    parser.add_argument("--assume-units", choices=("m", "cm", "mm"),
                        help="単位を推定せず指定する")
    parser.add_argument("--voxel-pitch", type=float,
                        help="ボクセル水密化の刻み(cm)。既定は最大辺/128")
    parser.add_argument("--out", help="書き出し先(既定: 入力名_clean.obj)")
    parser.add_argument("--selftest", action="store_true",
                        help="実スキャンなしで前処理の各段を自己テストする")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    elif args.mesh:
        out = args.out or (os.path.splitext(args.mesh)[0] + "_clean.obj")
        preprocess(args.mesh, assume_units=args.assume_units,
                   voxel_pitch=args.voxel_pitch, out_path=out)
    else:
        parser.error("mesh を指定するか --selftest を使ってください")
