"""
Phase 2 の土台となる3D幾何ヘルパー。

CLAUDE.md の設計判断に沿って:
  - 姿勢はオイラー角ではなくクォータニオンで持つ(ジンバルロックと
    補間の問題を避けるため)。scipyのRotation/Slerpに任せる。
  - 家具・階段の各ステップはすべて直方体(OBB: 中心・回転・半径ベクトル)
    として表す。凸分解などはせず、100行未満で済む形を優先する。
  - 「あと何cm」を返すため、衝突の有無だけでなく符号付き距離
    (signed distance)を返す関数を用意する。使うのは正確な分離軸
    定理(SAT)によるOBB間距離ではなく、Phase 1と同じ「形状の境界を
    サンプル点で近似し、各点から障害物集合までの距離の最小値を取る」
    やり方 -- 実装が簡単で、Phase 1のオラクルと設計が揃う。
"""

import numpy as np
from scipy.spatial.transform import Rotation


def random_quaternion(rng=None):
    """一様ランダムな3D回転をクォータニオン(x,y,z,w)で返す。

    オイラー角を単純に一様サンプルすると偏る(CLAUDE.mdで既知の罠として
    指摘されている)。scipyのRotation.randomは正しく一様分布を返す。
    """
    return Rotation.random(random_state=rng).as_quat()


def quat_slerp(q1, q2, t):
    """2つのクォータニオン間を球面線形補間(Slerp)する。t=0でq1、t=1でq2。"""
    key_rots = Rotation.from_quat([q1, q2])
    from scipy.spatial.transform import Slerp
    return Slerp([0, 1], key_rots)(t).as_quat()


def rotmat_from_quat(q):
    """クォータニオン(x,y,z,w) -> 3x3回転行列。"""
    return Rotation.from_quat(q).as_matrix()


def interpolate_se3(pos1, q1, pos2, q2, t):
    """位置は線形補間、姿勢はSlerpで、SE(3)状態を(pos, quat)として補間する。"""
    pos = pos1 + t * (pos2 - pos1)
    quat = quat_slerp(q1, q2, t)
    return pos, quat


def dist_se3(pos1, q1, pos2, q2, w=0.4):
    """SE(3)上の2状態間の「距離」(位置の差 + 回転角の差の重み付き和)。

    位置と回転は単位が違う(cmとラジアン)ので、そのまま足すと壊れる
    (CLAUDE.mdで最初に踏む落とし穴として明記されている罠そのもの)。
    wは家具の外接円半径くらいのオーダーが目安 -- 1ラジアン回すと、
    角がおおよそその半径ぶん動くため。
    """
    dp = np.linalg.norm(np.asarray(pos2) - np.asarray(pos1))
    r1 = Rotation.from_quat(q1)
    r2 = Rotation.from_quat(q2)
    dtheta = (r1.inv() * r2).magnitude()  # 回転角(ラジアン)
    return dp + w * dtheta


def obb_sdf(point, center, rotmat, half_extents):
    """点から直方体(OBB)までの符号付き距離。

    直方体のローカル座標系に点を変換してから、軸ごとに「半径をどれだけ
    超えているか」を見る、定番のboxのSDF公式(いわゆるInigo Quilez版):
    直方体の外側にいれば正の距離、内側にいればめり込み量を負の値で返す。
    """
    local = rotmat.T @ (np.asarray(point) - center)
    q = np.abs(local) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(float(np.max(q)), 0.0)
    return outside + inside


def obb_corners(center, rotmat, half_extents):
    """OBBの8つの角(ワールド座標)。描画・可視化用。"""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local = signs * half_extents
    return (local @ rotmat.T) + center


def box_surface_sample_points(half_extents, nu=5, nv=3, nw=3):
    """直方体の表面(6面)を覆う、ローカル座標でのサンプル点。

    Phase 1のbox_sample_pointsと同じ考え方: 角だけでなく面全体を
    サンプルすることで、対角線が障害物を突き抜けるケースも拾う。
    nu/nv/nwはそれぞれ長さ・幅・高さ方向の分割数。
    """
    hl, hw, hh = half_extents
    us = np.linspace(-hl, hl, nu)
    vs = np.linspace(-hw, hw, nv)
    ws = np.linspace(-hh, hh, nw)
    pts = []
    for u in us:
        for v in vs:
            pts.append((u, v, -hh))
            pts.append((u, v, hh))
    for u in us:
        for w in ws:
            pts.append((u, -hw, w))
            pts.append((u, hw, w))
    for v in vs:
        for w in ws:
            pts.append((-hl, v, w))
            pts.append((hl, v, w))
    return np.array(pts)


def cylinder_surface_sample_points(radius, half_height, n_theta=10, n_h=3):
    """縦向き円柱(運搬者の胴体)の表面のサンプル点(ローカル座標)。"""
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    hs = np.linspace(-half_height, half_height, n_h)
    pts = []
    for h in hs:
        for th in thetas:
            pts.append((radius * np.cos(th), radius * np.sin(th), h))
    return np.array(pts)
