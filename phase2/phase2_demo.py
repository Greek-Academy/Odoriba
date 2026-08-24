"""
Phase 2 最小デモ: 直線階段を6DOF(位置3 + クォータニオン4)の家具が
運べるか。Phase 1 のL字廊下を、3D・階段に一段階拡張したもの。

CLAUDE.md の Phase 2 方針:
  「性能が足りなくなってから OMPL / FCL を検討する。それまでは
  自作で通す。」

このファイルの読み方は phase1/phase1_demo.py と同じ構成:
  1. パラメータ(cm) -- 階段・家具・運搬者の寸法
  2. 環境 -- 階段を「外側の直方体(階段室)からステップの直方体群を
     くり抜いたもの」として表す(geometry3d.obb_sdfの上に構築)
  3. 家具・運搬者の表面サンプル点とクリアランス
  4. 可搬性オラクル(Phase 1と同じ「把持点+横位置探索」を3Dに拡張)
  5. RRT-Connect プランナー(6DOFなのでPhase 1のグリッドBFSは使えない
     -- 状態空間が広すぎて全探索が破綻する。ここは実際にRRTを使う)
  6. デモ / __main__
"""

import numpy as np
from scipy.spatial.transform import Rotation

import geometry3d as g3

# ---- パラメータ(cm) ----
# 階段: 蹴上げ(rise)・踏み面(tread)・段数・幅・踊り場。
# 実測前の仮値。Phase 0 の階段実測(measurement-protocol的な追加実測)
# ができたら、ここを実測値に置き換える。
STAIR_WIDTH = 90.0     # 階段の幅(x方向)
RISE = 17.0             # 1段の蹴上げ(z方向)
TREAD = 26.0            # 1段の踏み面奥行き(y方向)
N_STEPS = 8             # 段数
# 下/上の床の奥行き。家具(長辺180cm)がSTART/GOALの姿勢で水平に
# 収まり、かつ運搬者カプセルがCARRY_ARM+HUMAN_R分はみ出しても
# 床の端(壁ではなく単に部屋の境界)を越えない大きさにしてある
# (Phase 1のEND_MARGINと同じ考え方)。
BASE_LANDING_D = 300.0   # 下の(階段に入る前の)床の奥行き -- START用の余白
LANDING_D = 300.0        # 上の踊り場の奥行き -- GOAL用の余白
CEILING_CLEARANCE = 220.0  # 一番高い段の踏み面から天井までの高さ

# 家具(ソファを想定。implementation-guide.mdの出発点の数値と揃える)
FURN_L = 180.0   # 長辺
FURN_W = 60.0    # 短辺
FURN_H = 80.0    # 高さ

HUMAN_R = 20.0        # 運搬者の胴体半径(Phase 1と同じ、CLAUDE.md: 0.2m)
HUMAN_HEIGHT = 170.0  # 運搬者の身長
# 把持点から運搬者の体の中心までの水平方向の距離。Phase 1と同じ考え方。
CARRY_ARM = 30.0
# 運搬者が取り得る、家具の長軸に対する横方向の立ち位置(Phase 1と同じ)。
SIDE_OFFSETS = (-20.0, -10.0, 0.0, 10.0, 20.0)
# 把持点の高さから運搬者の体の中心までの鉛直オフセット。実際には足元は
# その場の段の高さで決まるが、正確な「どの段に足が乗るか」の計算は
# まだ実装していない -- 把持点付近に体があるという近似で済ませている。
# 段を上るときに縦方向の揺れがどれだけ効くかは未検証で、Phase 2 の
# 既知の簡略化点。結果が実測と大きくずれたらここを見直す。
GRIP_TO_TORSO_Z = -60.0

NUM_CARRIERS = 2


def build_stairs():
    """階段室を1つの外側の直方体として、そこから各ステップ(直方体)を
    くり抜いた環境を返す。

    戻り値: (outer, obstacles)
      outer = (center, rotmat, half_extents) -- 階段室全体の外枠
      obstacles = [(center, rotmat, half_extents), ...] -- 各ステップ
    """
    total_y = BASE_LANDING_D + N_STEPS * TREAD + LANDING_D
    total_z = N_STEPS * RISE + CEILING_CLEARANCE
    outer_half = np.array([STAIR_WIDTH / 2, total_y / 2, total_z / 2])
    outer_center = np.array([STAIR_WIDTH / 2, total_y / 2, total_z / 2])
    outer = (outer_center, np.eye(3), outer_half)

    obstacles = []
    for i in range(N_STEPS):
        # i段目: y方向は基準の床(BASE_LANDING_D)より先の
        # [BASE_LANDING_D + i*TREAD, BASE_LANDING_D + (i+1)*TREAD]、
        # その下(z=0から段の高さまで)がすべて実体(蹴込み板の後ろも
        # 含めてまとめて塞ぐ、簡略版)。
        step_h = RISE * (i + 1)
        y0 = BASE_LANDING_D + TREAD * i
        center = np.array([STAIR_WIDTH / 2, y0 + TREAD / 2, step_h / 2])
        half = np.array([STAIR_WIDTH / 2, TREAD / 2, step_h / 2])
        obstacles.append((center, np.eye(3), half))

    return outer, obstacles


def signed_clearance_3d(point, outer, obstacles):
    """点から環境境界までの符号付き距離(cm)。

    Phase 1のsigned_clearanceの3D版: 外枠の内側にいる余裕(+)から、
    各ステップにどれだけ食い込んでいるか(食い込んでいれば-)まで、
    一番厳しい値を返す。
    """
    outer_center, outer_rot, outer_half = outer
    # 外枠の内側にいる余裕 = -(外枠に対するSDF)。外側にいればマイナス
    # (どれだけはみ出しているか)、内側にいればプラス(壁までの距離)。
    margin = -g3.obb_sdf(point, outer_center, outer_rot, outer_half)
    worst = margin
    for center, rot, half in obstacles:
        d = g3.obb_sdf(point, center, rot, half)  # 障害物の外なら+、めり込んでいれば-
        worst = min(worst, d)
    return worst


def shape_clearance_3d(points, outer, obstacles):
    """点群(形状の表面サンプル)における最悪(最小)のクリアランス。"""
    return min(signed_clearance_3d(p, outer, obstacles) for p in points)


def furniture_world_points(pos, quat):
    """家具の表面サンプル点(ワールド座標)。"""
    half = np.array([FURN_L / 2, FURN_W / 2, FURN_H / 2])
    local = g3.box_surface_sample_points(half, nu=6, nv=3, nw=3)
    R = g3.rotmat_from_quat(quat)
    return (local @ R.T) + np.asarray(pos)


def furniture_clearance(pos, quat, outer, obstacles):
    """この姿勢での家具単体(運搬者なし)のクリアランス(cm)。"""
    return shape_clearance_3d(furniture_world_points(pos, quat), outer, obstacles)
