"""
運搬者の人型メッシュ(描画用): 球・円柱・楕円体を組み合わせて、
頭・胴・腕・脚のある人の形をコードで作る。

衝突判定は今のところ phase2_lstair の円柱(carrier_points)のまま。
ここは見た目だけを担当する。ただし体の寸法はこのモジュールの定数に
集めておき、判定を人型にするときも同じ値を使う(寸法の出どころを1つにする)。

寸法はいずれも実測なしの常識的な仮値(cm、身長170cmの成人)。
"""

import numpy as np

# ---- 体の寸法(cm) ----
HEIGHT = 170.0
HEAD_R = 11.0          # 頭の半径
SHOULDER_Z = 145.0     # 肩の高さ(足元から)
SHOULDER_HW = 20.0     # 肩幅の半分
TORSO_HD = 12.0        # 胴の前後方向の厚みの半分
WAIST_HW = 16.0        # 腰回りの横幅の半分
HIP_Z = 90.0           # 股関節の高さ
HIP_HW = 10.0          # 左右の股関節の間隔の半分
LEG_R = 7.0            # 脚の半径
NECK_R = 5.0
ARM_R = 5.0            # 腕の半径
UPPER_ARM = 30.0       # 上腕の長さ
FOREARM = 28.0         # 前腕(手首まで)の長さ
HAND_R = 5.0

# メッシュの分割数。経路の全フレーム分を HTML に埋め込むため、見た目が
# 崩れない範囲で粗くしてファイルサイズを抑える。
N_RING = 8             # 円柱・球の周方向
N_LAT = 5              # 球の緯度方向


# ---- 単位形状 ----

def _unit_sphere(n_lat=N_LAT, n_lon=N_RING):
    """原点中心・半径1の球(両極+緯度リング)。"""
    verts = [(0.0, 0.0, -1.0)]
    for i in range(1, n_lat):
        phi = -np.pi / 2 + np.pi * i / n_lat
        for j in range(n_lon):
            th = 2 * np.pi * j / n_lon
            verts.append((np.cos(phi) * np.cos(th), np.cos(phi) * np.sin(th), np.sin(phi)))
    verts.append((0.0, 0.0, 1.0))
    top = len(verts) - 1

    def ring(i, j):  # i番目(1始まり)の緯度リングのj番目の頂点
        return 1 + (i - 1) * n_lon + j % n_lon

    tris = []
    for j in range(n_lon):
        tris.append((0, ring(1, j + 1), ring(1, j)))
        tris.append((top, ring(n_lat - 1, j), ring(n_lat - 1, j + 1)))
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            tris.append((a, b, d))
            tris.append((a, d, c))
    return np.array(verts), np.array(tris)


def _unit_cylinder(n=N_RING):
    """z=0..1・半径1の円柱の側面(蓋なし。端は球や楕円体でふさぐ)。"""
    th = 2 * np.pi * np.arange(n) / n
    ring = np.column_stack([np.cos(th), np.sin(th)])
    verts = np.vstack([np.column_stack([ring, np.zeros(n)]),
                       np.column_stack([ring, np.ones(n)])])
    tris = []
    for a in range(n):
        b = (a + 1) % n
        tris.append((a, b, n + a))
        tris.append((b, n + b, n + a))
    return verts, np.array(tris)


_SPHERE = _unit_sphere()
_CYLINDER = _unit_cylinder()


def _transform(unit, M, origin):
    """単位形状を線形変換M(列が各軸のベクトル)と平行移動で配置する。"""
    v, t = unit
    return v @ np.asarray(M).T + np.asarray(origin), t


def _perp_basis(axis):
    """axisに直交する単位ベクトル2本。"""
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


# ---- 部品 ----

def sphere(center, r):
    return _transform(_SPHERE, np.eye(3) * r, center)


def ellipsoid(center, ax0, ax1, ax2):
    """3本の半軸ベクトル(互いに直交、長さ=半径)で決まる楕円体。"""
    return _transform(_SPHERE, np.column_stack([ax0, ax1, ax2]), center)


def capsule(a, b, r, cap_a=True, cap_b=True):
    """点aから点bまでの半径rの円柱+端の球。

    端がほかの部品(胴・手の球など)に埋もれる場合はcap_a/cap_bをFalseに
    して頂点数を減らす。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    if np.linalg.norm(axis) < 1e-6:
        return sphere(a, r)
    u, w = _perp_basis(axis)
    parts = [_transform(_CYLINDER, np.column_stack([u * r, w * r, axis]), a)]
    if cap_a:
        parts.append(sphere(a, r))
    if cap_b:
        parts.append(sphere(b, r))
    return merge(parts)


def merge(parts):
    """複数の(verts, tris)を1つにまとめる(三角形の番号をずらして連結)。"""
    verts, tris, offset = [], [], 0
    for v, t in parts:
        verts.append(v)
        tris.append(t + offset)
        offset += len(v)
    return np.vstack(verts), np.vstack(tris)


# ---- 腕の簡易IK ----

def elbow_position(shoulder, hand, outward):
    """肩とhandを結ぶ2リンク(上腕・前腕)の肘の位置。

    肘は下向き+体の外側(outward)に曲げる。手が届かない距離なら
    肩から手へまっすぐ伸ばした線上に置く(描画専用なので、届かない分は
    腕が伸びて見えるだけ。手が届くかの判定はオラクル側のreach項が持つ)。
    """
    d_vec = hand - shoulder
    d = np.linalg.norm(d_vec)
    if d < 1e-6:
        return shoulder + np.array([0.0, 0.0, -UPPER_ARM])
    direction = d_vec / d
    if d >= UPPER_ARM + FOREARM:
        return shoulder + direction * d * UPPER_ARM / (UPPER_ARM + FOREARM)
    d = max(d, abs(UPPER_ARM - FOREARM) + 1e-3)
    along = (UPPER_ARM ** 2 - FOREARM ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(UPPER_ARM ** 2 - along ** 2, 0.0))
    pole = np.array([0.0, 0.0, -1.0]) + 0.6 * outward
    perp = pole - direction * (pole @ direction)
    if np.linalg.norm(perp) < 1e-6:
        perp = outward - direction * (outward @ direction)
    perp /= np.linalg.norm(perp)
    return shoulder + direction * along + perp * h


# ---- 1人分 ----

def figure_mesh(foot_xy, floor_z, facing_xy, hands):
    """足元(foot_xy, floor_z)に立ち、facing_xyの方向を向いて、
    両手をhands(左右の手先の目標点2つ、ワールド座標)に伸ばした人の
    メッシュ(verts, tris)を返す。"""
    f = np.array([facing_xy[0], facing_xy[1], 0.0], dtype=float)
    f /= np.linalg.norm(f)
    s = np.array([-f[1], f[0], 0.0])        # 体の左方向
    z = np.array([0.0, 0.0, 1.0])
    base = np.array([foot_xy[0], foot_xy[1], floor_z], dtype=float)

    def at(height, side=0.0, fwd=0.0):
        return base + z * height + s * side + f * fwd

    parts = []
    # 脚: 股関節から足首まで
    for side in (HIP_HW, -HIP_HW):
        parts.append(capsule(at(HIP_Z, side), at(LEG_R, side), LEG_R, cap_a=False))
    # 胴: 腰(楕円体)から肩(楕円体)までを楕円柱でつなぐ。肩幅が広く前後は薄い
    chest_z = SHOULDER_Z - 6.0
    parts.append(_transform(_CYLINDER,
                            np.column_stack([f * TORSO_HD, s * WAIST_HW,
                                             z * (chest_z - HIP_Z)]),
                            at(HIP_Z)))
    parts.append(ellipsoid(at(HIP_Z), f * TORSO_HD, s * WAIST_HW, z * 8.0))
    parts.append(ellipsoid(at(chest_z), f * TORSO_HD, s * SHOULDER_HW, z * 8.0))
    # 首と頭
    head_c = at(HEIGHT - HEAD_R)
    parts.append(capsule(at(SHOULDER_Z), head_c - z * HEAD_R, NECK_R,
                         cap_a=False, cap_b=False))
    parts.append(sphere(head_c, HEAD_R))
    # 腕: 肩 -> 肘 -> 手。左寄りの目標点を左手に割り当て、腕が交差しないようにする
    hands = sorted((np.asarray(h, dtype=float) for h in hands),
                   key=lambda h: -((h - base) @ s))
    for sign, hand in zip((1.0, -1.0), hands):
        shoulder = at(SHOULDER_Z - ARM_R, sign * (SHOULDER_HW - ARM_R))
        hand = np.asarray(hand, dtype=float)
        elbow = elbow_position(shoulder, hand, s * sign)
        parts.append(capsule(shoulder, elbow, ARM_R))
        parts.append(capsule(elbow, hand, ARM_R * 0.85, cap_a=False, cap_b=False))
        parts.append(sphere(hand, HAND_R))
    return merge(parts)
