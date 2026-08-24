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
CEILING_CLEARANCE = 160.0  # 一番高い段の踏み面から天井までの高さ(古い建物の低い天井を想定)

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
# 把持点の高さから運搬者の体の中心までの鉛直オフセット。人は把持点
# より高い位置に胴があるので正の値(把持点が低ければ屈んで持つ)。
# 実際には足元はその場の段の高さで決まるが、正確な「どの段に足が
# 乗るか」の計算はまだ実装していない -- 把持点付近に体があるという
# 近似で済ませている。段を上るときに縦方向の揺れがどれだけ効くかは
# 未検証で、Phase 2 の既知の簡略化点。結果が実測と大きくずれたら
# ここを見直す。
GRIP_TO_TORSO_Z = 50.0

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


def place_humans_3d(pos, quat, side_offset, num_carriers=None):
    """与えられた横方向の立ち位置に対する、運搬者カプセルの中心。

    Phase 1のplace_humansをそのまま3Dに拡張したもの: 把持点は家具の
    長軸(ローカルx)の両端に置き、そこからCARRY_ARMだけ長軸方向に
    外側へ、side_offsetだけ短軸(ローカルy)方向へずらす。鉛直方向
    (ローカルz、家具が傾いていてもワールドzではなく家具のローカルz
    を使う)には、GRIP_TO_TORSO_Zだけ下げて運搬者の胴体中心とする
    -- 「手の高さのやや下に胴の中心がある」という近似(実測なし、
    モジュールdocstring/GRIP_TO_TORSO_Zの説明を参照)。

    num_carriers=2 (既定): 両端に1人ずつ。
    num_carriers=1: 家具の後方(ローカルxが負の側)を1人で持つ。
    """
    num_carriers = NUM_CARRIERS if num_carriers is None else num_carriers
    R = g3.rotmat_from_quat(quat)
    pos = np.asarray(pos)
    hl = FURN_L / 2

    def grip_to_world(local_x_sign):
        local = np.array([
            local_x_sign * (hl + CARRY_ARM),
            side_offset,
            GRIP_TO_TORSO_Z,
        ])
        return pos + R @ local

    h_back = grip_to_world(-1.0)
    if num_carriers == 1:
        return (h_back,)
    h_front = grip_to_world(1.0)
    return (h_front, h_back)


def carrier_world_points(center):
    """運搬者カプセル(縦向き円柱)の表面サンプル点(ワールド座標)。

    円柱はワールド座標系で鉛直に立てる(家具が傾いていても、人は
    重力に従って立つため、家具のローカル姿勢では回転させない)。
    """
    local = g3.cylinder_surface_sample_points(HUMAN_R, HUMAN_HEIGHT / 2, n_theta=10, n_h=3)
    return local + np.asarray(center)


def carriable_clearance(pos, quat, outer, obstacles, num_carriers=None):
    """許容される運搬者の立ち位置全体の中で達成できる最良のクリアランス(cm)。

    Phase 1のcarriable_clearanceと同じ考え方: SIDE_OFFSETSを総当たりし、
    一番有利なものを残す。
    """
    bc = furniture_clearance(pos, quat, outer, obstacles)
    if bc < 0:
        return bc, None
    best = -np.inf
    best_offset = None
    for off in SIDE_OFFSETS:
        centers = place_humans_3d(pos, quat, off, num_carriers)
        capsule_clearances = [
            shape_clearance_3d(carrier_world_points(c), outer, obstacles)
            for c in centers
        ]
        cand = min([bc] + capsule_clearances)
        if cand > best:
            best = cand
            best_offset = off
    return best, best_offset


def best_human_positions(pos, quat, outer, obstacles, num_carriers=None):
    """描画専用: 最良の立ち位置での運搬者カプセルの中心。"""
    _, offset = carriable_clearance(pos, quat, outer, obstacles, num_carriers=num_carriers)
    if offset is None:
        offset = 0.0
    return place_humans_3d(pos, quat, offset, num_carriers)


def collision_free(pos, quat, with_human, outer, obstacles, num_carriers=None):
    """プランナーが各状態に対して使う、唯一のYes/No判定(Phase 1と同じ役割)。"""
    if with_human:
        clearance, _ = carriable_clearance(pos, quat, outer, obstacles, num_carriers)
        return clearance >= 0
    return furniture_clearance(pos, quat, outer, obstacles) >= 0


# ---- RRT-Connect プランナー ----
# Phase 1はグリッド上のBFSで全探索できたが、ここは(位置3 + 姿勢4の)
# 6DOF。グリッドを細かく切って全探索すると状態数が現実的な時間で
# 収まらない。CLAUDE.mdの通りRRT-Connectを自作する。

W_ROT = 0.4  # dist_se3の回転項の重み。家具の外接半径くらいのオーダーが目安
MAX_STEP = 20.0  # 1回のextendで進む最大距離(cm、dist_se3換算)
EDGE_RES = 4.0    # エッジ検証(衝突チェック)の刻み幅(cm)
GOAL_TOL_POS = 8.0    # ゴール判定: 位置の許容誤差(cm)
GOAL_TOL_ROT = 0.15   # ゴール判定: 姿勢の許容誤差(ラジアン)


STAIR_ANGLE = np.arctan2(RISE, TREAD)  # 階段の傾斜角(ラジアン)


def stairs_profile_z(y):
    """階段の中心線に沿って、その y 座標での「歩行面の高さ」を返す。
    sample_guidedが位置のサンプリングを誘導するために使うだけの
    大まかな目安で、衝突判定には使わない。"""
    if y < BASE_LANDING_D:
        return 0.0
    step_y = y - BASE_LANDING_D
    i = min(int(step_y // TREAD), N_STEPS - 1)
    if step_y >= N_STEPS * TREAD:
        return N_STEPS * RISE
    return RISE * (i + 1)


def _base_yaw_quat():
    """家具の長軸(ローカルx)を進行方向(ワールドy)に向ける基準姿勢。"""
    return Rotation.from_euler("z", 90, degrees=True)


def sample_guided(outer, rng, p=0.7, pos_noise=15.0, rot_noise_deg=10.0):
    """状態空間からのサンプリング。

    狭い通路(踊り場・階段)ではランダムサンプリングがほぼ当たらない
    (CLAUDE.mdの「骨格でサンプリングを誘導する」対策そのもの)。

    位置だけでなく姿勢の誘導も要る。この家具(長辺180cm)は踏み面
    26cmよりずっと長いため、水平のまま(傾けずに)階段区間に置くと
    ほぼ必ずどこかの段にぶつかる -- 現実に階段で家具を運ぶときに
    傾けるのと同じ理由。そこで、階段区間(y方向)をサンプルしたときは
    長軸を進行方向に向けたうえで、階段の傾斜角(STAIR_ANGLE)を中心に
    ノイズを乗せてピッチさせる。踊り場区間では傾斜0を中心にする。
    確率(1-p)では姿勢・位置とも完全ランダムにし、この誘導だけでは
    見つからない解(踊り場での回転など)も探索できるようにする。
    """
    outer_center, _, outer_half = outer
    if rng.random() < p:
        y = rng.uniform(0.0, outer_half[1] * 2)
        z = stairs_profile_z(y) + FURN_H / 2
        x = outer_half[0] + rng.normal(0.0, pos_noise * 0.5)
        pos = np.array([x, y, z]) + rng.normal(0.0, pos_noise, size=3) * np.array([1, 1, 0.3])

        on_stairs = BASE_LANDING_D <= y <= BASE_LANDING_D + N_STEPS * TREAD
        target_pitch = STAIR_ANGLE if on_stairs else 0.0
        pitch = target_pitch + np.radians(rng.normal(0.0, rot_noise_deg))
        yaw_noise = np.radians(rng.normal(0.0, rot_noise_deg))
        roll_noise = np.radians(rng.normal(0.0, rot_noise_deg))
        rot = (Rotation.from_euler("z", 90 + np.degrees(yaw_noise), degrees=True)
               * Rotation.from_euler("x", np.degrees(pitch), degrees=True)
               * Rotation.from_euler("y", np.degrees(roll_noise), degrees=True))
        quat = rot.as_quat()
    else:
        pos = rng.uniform(0.0, outer_half * 2)
        quat = g3.random_quaternion(rng)
    return pos, quat


def edge_valid(pos1, quat1, pos2, quat2, with_human, outer, obstacles, num_carriers=None):
    """2状態間を補間し、刻み幅EDGE_RESごとに衝突チェックする。

    端点だけ見ると、壁を突き抜ける経路が「有効」と判定されてしまう
    (Phase 1のimplementation-guide.mdで最初に触れている罠と同じ)。
    """
    d = g3.dist_se3(pos1, quat1, pos2, quat2, w=W_ROT)
    n = max(2, int(d / EDGE_RES))
    for i in range(n + 1):
        t = i / n
        p, q = g3.interpolate_se3(pos1, quat1, pos2, quat2, t)
        if not collision_free(p, q, with_human, outer, obstacles, num_carriers):
            return False
    return True


class _Node:
    __slots__ = ("pos", "quat", "parent")

    def __init__(self, pos, quat, parent=None):
        self.pos = pos
        self.quat = quat
        self.parent = parent


def _nearest(tree, pos, quat):
    dists = [g3.dist_se3(n.pos, n.quat, pos, quat, w=W_ROT) for n in tree]
    i = int(np.argmin(dists))
    return i, dists[i]


def _steer(pos_from, quat_from, pos_to, quat_to):
    """pos_from/quat_fromから、pos_to/quat_toの方向へMAX_STEPだけ進んだ状態。"""
    d = g3.dist_se3(pos_from, quat_from, pos_to, quat_to, w=W_ROT)
    t = 1.0 if d <= MAX_STEP else MAX_STEP / d
    return g3.interpolate_se3(pos_from, quat_from, pos_to, quat_to, t)


def _extend(tree, pos_target, quat_target, with_human, outer, obstacles, num_carriers):
    """treeを1歩だけpos_target/quat_targetへ伸ばす。伸びたら新しいノード
    のインデックスを、伸びなければNoneを返す。"""
    i_near, _ = _nearest(tree, pos_target, quat_target)
    near = tree[i_near]
    pos_new, quat_new = _steer(near.pos, near.quat, pos_target, quat_target)
    if not edge_valid(near.pos, near.quat, pos_new, quat_new, with_human, outer, obstacles, num_carriers):
        return None
    tree.append(_Node(pos_new, quat_new, i_near))
    return len(tree) - 1


def _connect(tree, pos_target, quat_target, with_human, outer, obstacles, num_carriers):
    """targetに向かって、ブロックされるかtargetに届くまでextendを繰り返す
    (RRT-Connectの"connect"ヒューリスティック: 1本のツリーを毎回1歩ずつ
    伸ばすより、狭い通路を素早く抜けやすい)。"""
    last = None
    while True:
        idx = _extend(tree, pos_target, quat_target, with_human, outer, obstacles, num_carriers)
        if idx is None:
            return last
        last = idx
        node = tree[idx]
        if g3.dist_se3(node.pos, node.quat, pos_target, quat_target, w=W_ROT) < 1e-6:
            return last  # targetそのものに到達


def _reached(node, pos, quat):
    dp = np.linalg.norm(node.pos - pos)
    r1 = Rotation.from_quat(node.quat)
    r2 = Rotation.from_quat(quat)
    dtheta = (r1.inv() * r2).magnitude()
    return dp < GOAL_TOL_POS and dtheta < GOAL_TOL_ROT


def _build_path(tree_a, idx_a, tree_b, idx_b, swapped):
    """2本のツリーが繋がった点から、start->goalの状態列を組み立てる。"""
    path_a = []
    i = idx_a
    while i is not None:
        n = tree_a[i]
        path_a.append((n.pos, n.quat))
        i = n.parent
    path_a.reverse()

    path_b = []
    i = idx_b
    while i is not None:
        n = tree_b[i]
        path_b.append((n.pos, n.quat))
        i = n.parent

    if swapped:
        path_a, path_b = path_b, path_a
    return path_a + path_b


def rrt_connect(start, goal, with_human, outer, obstacles, num_carriers=None,
                 max_iter=3000, seed=0):
    """RRT-Connect本体。startとgoalそれぞれからツリーを伸ばし、交互に
    相手のツリーの新しいノードへconnectを試みる(implementation-guide.md
    のrrt_connectと同じ構造)。

    見つかればstart->goalの(pos, quat)状態列を、見つからなければ
    Noneを返す。
    """
    start_pos, start_quat = start
    goal_pos, goal_quat = goal
    if not collision_free(start_pos, start_quat, with_human, outer, obstacles, num_carriers):
        return None
    if not collision_free(goal_pos, goal_quat, with_human, outer, obstacles, num_carriers):
        return None

    ta = [_Node(np.asarray(start_pos, dtype=float), np.asarray(start_quat, dtype=float))]
    tb = [_Node(np.asarray(goal_pos, dtype=float), np.asarray(goal_quat, dtype=float))]
    swapped = False
    rng = np.random.default_rng(seed)

    for _ in range(max_iter):
        pos_rand, quat_rand = sample_guided(outer, rng)
        idx_new = _extend(ta, pos_rand, quat_rand, with_human, outer, obstacles, num_carriers)
        if idx_new is not None:
            new_node = ta[idx_new]
            idx_conn = _connect(tb, new_node.pos, new_node.quat, with_human, outer, obstacles, num_carriers)
            if idx_conn is not None and _reached(tb[idx_conn], new_node.pos, new_node.quat):
                return _build_path(ta, idx_new, tb, idx_conn, swapped)
        ta, tb = tb, ta
        swapped = not swapped

    return None
