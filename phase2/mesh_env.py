"""
スキャンメッシュの受け口: 環境を三角形メッシュとして受け取り、
点群の符号付きクリアランス(cm)を返す。

iPhone LiDAR のスキャン(OBJ/PLY)を判定に使うための入り口。CLAUDE.md の
「入力層は差し替え可能にする」の実装で、環境の表現が

  - 既存: 直方体の集まり(build_lstairsなど) -> 解析的なSDF
  - 本モジュール: 自由空間を表す水密メッシュ -> trimeshの符号付き距離
    (メッシュの内側 = 自由空間 = 正)

のどちらでも、プランナー(rrt_connect)側は validator を1個すげ替える
だけで同じに動く。

スマホ入手前の検証として、直方体環境から同じ自由空間のメッシュを
ブーリアン演算で合成できる(env_to_mesh)。実スキャンが来たら
trimesh.load でメッシュを読み、MeshEnv に渡すだけで以降は共通
(スキャンは「部屋の内面」の面メッシュなので、穴埋め・水密化の
前処理が必要になる見込み。それはスキャン実物を見てから決める)。

実行(直方体方式との判定一致チェック):
    python mesh_env.py
"""

import numpy as np
import trimesh

import phase2_lstair as L


def _box_mesh(center, half):
    """軸平行の直方体(中心・半径ベクトル)をtrimeshのboxにする。"""
    m = trimesh.creation.box(extents=np.asarray(half) * 2)
    m.apply_translation(np.asarray(center))
    return m


def env_to_mesh(outer, obstacles):
    """直方体環境(外枠の合併 - 障害物)から自由空間の水密メッシュを合成する。

    ブーリアンは manifold エンジン(正確・水密前提)を使う。障害物が
    外枠の面とぴったり重なる(共平面)箇所が多いが、manifoldは厳密演算
    なので問題なく処理できる。
    """
    free = trimesh.boolean.union([_box_mesh(c, h) for c, _, h in outer],
                                 engine="manifold")
    solid = trimesh.boolean.union([_box_mesh(c, h) for c, _, h in obstacles],
                                  engine="manifold")
    return trimesh.boolean.difference([free, solid], engine="manifold")


class MeshEnv:
    """メッシュ環境に対する符号付きクリアランスの計算器。

    mesh は「自由空間」を表す水密メッシュ(内側にいる = ぶつかっていない)。
    """

    def __init__(self, mesh, require_watertight=True):
        # 水密なら符号付き距離(内外の符号)が信用できる。実スキャンは
        # 完全な水密化に失敗することがあるので、require_watertight=Falseなら
        # 警告だけ出して先に進み、符号なし距離(表面までの距離)にフォール
        # バックする。フォールバックでは内外の区別ができないため、
        # クリアランスは常に非負になり「めり込み量(負)」は出せない
        # (まず判定を回して形を確認するための応急モード)。
        self.watertight = bool(mesh.is_watertight)
        if require_watertight and not self.watertight:
            raise ValueError("自由空間メッシュが水密でない(符号が信用できない)。"
                             "require_watertight=Falseで応急的に符号なし距離を使える")
        if not self.watertight:
            print("警告: メッシュが水密でない。符号なし距離にフォールバックする"
                  "(内外の区別ができず、めり込み量は0でクリップされる)")
        self.mesh = mesh
        self._pq = trimesh.proximity.ProximityQuery(mesh)

    def signed_clearance_points(self, points):
        """各点の符号付きクリアランス(自由空間の内側なら正)。

        水密でない場合は符号が付かないので、表面までの距離(非負)を返す。
        """
        pts = np.asarray(points)
        if self.watertight:
            return self._pq.signed_distance(pts)
        return self._pq.distance(pts)

    def shape_clearance(self, points):
        """点群(形状の表面サンプル)の最悪クリアランス。
        phase2_demo.shape_clearance_3d と同じ役割のメッシュ版。"""
        return float(np.min(self.signed_clearance_points(points)))

    def furniture_clearance(self, pos, quat):
        """家具単体のクリアランス。phase2_lstair.furniture_clearanceのメッシュ版。"""
        return self.shape_clearance(L.furniture_points(pos, quat))

    def validator(self, pos, quat, with_human, outer, obstacles, num_carriers=None):
        """rrt_connect にそのまま渡せる判定関数(collision_freeと同じ形)。

        outer/obstacles はシグネチャ互換のため受け取るが使わない
        (環境はメッシュ側が持っている)。運搬者ありの保持拘束
        (傾き上限・床の上に立つ・手の届く範囲)は phase2_lstair の
        オラクルと同じロジックで、距離計算だけメッシュに差し替える。
        """
        if not with_human:
            return self.furniture_clearance(pos, quat) >= 0
        if L.with_tilt_violation(quat):
            return False
        if self.furniture_clearance(pos, quat) < 0:
            return False
        for off in L.p.SIDE_OFFSETS:
            placed = L.place_humans_lstair(pos, quat, off, num_carriers)
            if placed is None:
                continue
            ok = all(reach >= 0 and
                     self.shape_clearance(L.carrier_points(c)) >= 0
                     for c, reach in placed)
            if ok:
                return True
        return False


# ---- 直方体方式との一致チェック ----

if __name__ == "__main__":
    import json
    import time

    outer, obstacles = L.build_lstairs()
    print("自由空間メッシュを合成中(ブーリアン)...")
    t0 = time.time()
    mesh = env_to_mesh(outer, obstacles)
    print(f"  watertight={mesh.is_watertight}, faces={len(mesh.faces)}, "
          f"volume={mesh.volume / 1e6:.1f} m^3 [{time.time() - t0:.1f}s]")
    env = MeshEnv(mesh)

    out_obj = "lstair_env.obj"
    mesh.export(out_obj)
    print(f"  メッシュを {out_obj} に書き出した(git管理外。LiDARスキャンの代役)")

    # 1) 保存済みの家具単体経路: 直方体方式で有効だった全状態が、
    #    メッシュ判定でも有効(クリアランス>=0)になるか
    with open("results/lstair_result.json") as f:
        result = json.load(f)
    path = [(np.array(s[:3]), np.array(s[3:])) for s in result["box_only"]["path"]]
    t0 = time.time()
    diffs = []
    n_ok = 0
    for pos, quat in path:
        c_box = L.furniture_clearance(pos, quat, outer, obstacles)
        c_mesh = env.furniture_clearance(pos, quat)
        # 直方体方式は外枠同士の重なり付近で距離を過小評価する近似
        # (shape_clearance_3dのdocstring)なので、c_mesh >= c_box が期待値
        diffs.append(c_mesh - c_box)
        if c_mesh >= -1e-6:
            n_ok += 1
    print(f"\n保存済み経路({len(path)}状態): メッシュ判定でも有効 {n_ok}/{len(path)} "
          f"[{time.time() - t0:.1f}s]")
    print(f"  クリアランス差(mesh - box): min={min(diffs):.2f} / max={max(diffs):.2f}cm "
          "(負が大きいと不一致、正は重なり付近の過小評価が解けた分)")

    # 2) ランダム姿勢での衝突判定(True/False)の一致率
    rng = np.random.default_rng(7)
    sampler = L.make_sampler(with_human=False)
    n_trial, n_agree, worst = 200, 0, 0.0
    t0 = time.time()
    for _ in range(n_trial):
        pos, quat = sampler(rng)
        c_box = L.furniture_clearance(pos, quat, outer, obstacles)
        c_mesh = env.furniture_clearance(pos, quat)
        if (c_box >= 0) == (c_mesh >= 0):
            n_agree += 1
        elif abs(c_box) > worst:
            worst = abs(c_box)
    print(f"\nランダム姿勢{n_trial}件の衝突判定一致: {n_agree}/{n_trial} "
          f"(不一致の最大クリアランス絶対値 {worst:.2f}cm) [{time.time() - t0:.1f}s]")
    print("\n判定: 一致率が~100%なら、LiDARスキャン(水密化済みメッシュ)を"
          "load->MeshEnvに渡すだけで既存プランナーがそのまま使える。")
