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


def to_free_space_voxel(mesh, pitch=6.0, seal_iters=1, min_vol_m3=0.5):
    """スキャン(壁・段の面)から「囲まれた空気の部分=自由空間」の水密
    メッシュを取り出す。

    手順: ボクセル化 -> 面を seal_iters 回だけ膨張(binary_dilation)して
    小さな穴を塞ぐ -> 外周に空きの余白を足して「外側」を作る ->
    binary_fill_holes で外側から届かない空きボクセル(=囲まれた空気)を
    取り出す -> marching cubes で閉じた面にする。

    膨張(dilation)であって膨張収縮(closing)ではない点が要注意: closingは
    内部の大きな空洞まで潰してしまい、閉じた入力でも「空気ゼロ」と誤判定
    する(検証で確認済み)。膨張は壁を厚くして小穴を塞ぐだけなので、空洞は
    残る(そのぶん内部が seal_iters ボクセル分だけ小さめに出る)。

    重要な前提チェック: スキャンが壁を十分に囲えていないと、内側の空気が
    外側と繋がっていて「囲まれた空気」がほとんど無い(min_vol_m3未満)。その
    場合は水密化できない(Noneと理由を返す)。より完全なスキャンか、手動での
    蓋付け、または実測寸法での箱モデル(プランB)が必要というシグナル。
    戻り値: (mesh or None, message)
    """
    from scipy import ndimage
    vg = mesh.voxelized(pitch=pitch)
    occ = np.asarray(vg.matrix, dtype=bool)
    if seal_iters:
        occ = ndimage.binary_dilation(occ, iterations=seal_iters)
    # 外周に空きの余白を1層足して、確実に「外側の空き」を作る
    occ = np.pad(occ, 1, mode="constant", constant_values=False)
    # 外側から届かない空き = 囲まれた空気(enclosed pocket)
    filled = ndimage.binary_fill_holes(occ)
    interior = filled & ~occ
    interior = interior[1:-1, 1:-1, 1:-1]  # 余白を戻す

    vol_m3 = interior.sum() * pitch ** 3 / 1e6
    if vol_m3 < min_vol_m3:
        return None, (f"囲まれた空気がほとんど無い({vol_m3:.2f} m^3)=壁が閉じて"
                      "いない部分スキャン。より完全なスキャンか手動蓋付けが必要")

    free_vg = trimesh.voxel.VoxelGrid(interior, transform=vg.transform)
    mc = free_vg.marching_cubes
    trimesh.repair.fix_normals(mc)
    return mc, f"囲まれた自由空間 {vol_m3:.1f} m^3 (pitch={pitch}cm)"


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

    # (d) 囲まれた空間の抽出: 閉じた箱は空気が出る/一面開けた箱はNone
    import trimesh as tm
    room = tm.creation.box(extents=[300, 200, 240])
    mroom, msg_r = to_free_space_voxel(room, pitch=8.0)
    ok_closed = mroom is not None and mroom.is_watertight
    op = room.copy()
    xmax = room.vertices[:, 0].max()
    op.update_faces(~np.all(room.vertices[:, 0][op.faces] > xmax - 1e-6, axis=1))
    mopen, msg_o = to_free_space_voxel(op, pitch=8.0)
    ok_open = mopen is None
    print(f"(d) 囲まれた空間の抽出: 閉じた箱={'OK' if ok_closed else 'NG'}({msg_r}) / "
          f"一面開け={'OK(None)' if ok_open else 'NG'}")


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
