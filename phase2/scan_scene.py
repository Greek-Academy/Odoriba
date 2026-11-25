"""
判定の結果を、スキャンの立体表示(scan_picker の画面)に重ねるための形にする。

scan_plan.judge の結果(運搬者ありの掃引)から、経路上の各位置での家具と
運搬者の形を、頂点の並びとして取り出す。座標は、立体表示が読んだ元のファイルの
座標に直して渡す(画面側は受け取った頂点をそのまま描くだけにして、回転の
向きの取り決めを画面側に持ち込まない)。

各位置の姿勢は「その位置で、試した姿勢の中で一番余裕のあるもの」。位置と
位置の間をどう動かすかは掃引では確かめていないので、並べて見せるときは
そのことを画面に書く。
"""

import itertools

import numpy as np

import human_figure as hf
import phase2_lstair as L
import phase2_lstair_3d as V

# 直方体の8つの角(_box_corners の並び)を結ぶ三角形
_BOX_QUADS = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
BOX_TRIS = [i for a, b, c, d in _BOX_QUADS for i in (a, b, c, a, c, d)]
_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


def _box_corners(pos, quat):
    """家具の8つの角(判定側の座標)。"""
    center, rot, half = V.furniture_state(pos, quat)
    return center + (_SIGNS * half) @ rot.T


def _people(space, pos, quat, num_carriers):
    """運搬者それぞれの (頂点, 三角形)。"""
    cl = space.centerline
    out = []
    for cp in L.carrier_poses_lstair(pos, quat, num_carriers, space, None, space.floor_fn):
        # 立つ床がない所(壁の外など)では、宙に浮かせず近くの中心線の床に
        # 立たせる。壁を突き抜けて見えるので「ここには立てない」と分かる
        x, y = cp["foot_xy"]
        floor_z = cp["floor_z"]
        if space.floor_fn(x, y, cp["center"][2]) is None:
            floor_z = cl.z_at(cl.project(x, y))
        out.append(hf.figure_mesh(cp["foot_xy"], floor_z, cp["facing"], cp["hands"]))
    return out


def result_scene(space, result, num_carriers, from_zup):
    """運搬者ありの掃引の結果を、立体表示に重ねる dict にする。

    家具の寸法(phase2_lstair の変数)が判定のときと同じ状態で呼ぶこと。
    from_zup: 判定側の座標(上がZ・cm)を、立体表示のファイルの座標に直す4×4行列。

    戻り値:
      frames: 経路上の位置ごとに {s, clearance, at, box, people}。at は中心線上の
              その位置(床の少し上)。持てる姿勢が無い位置は clearance・box・people が None
      box_tris / people_tris: 三角形の頂点番号(全ての位置で共通)
      bottleneck: 一番狭い所の {frame, capacity_cm, contact, part}
      furniture_only_cm: 家具だけ(人なし)での一番狭い所の余裕
      length: 中心線の長さ(cm)
    座標は [x, y, z, x, y, z, ...] と1列に並べる。
    """
    M = np.asarray(from_zup, dtype=float)

    def flat(pts):
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        return np.round(p @ M[:3, :3].T + M[:3, 3], 4).ravel().tolist()

    cl = space.centerline
    b = result["bottleneck"]
    frames, people_tris = [], None
    for (s, c), pose in zip(b["profile"], b["poses"]):
        frame = {"s": float(s), "clearance": None, "box": None, "people": None,
                 "at": flat([*cl.xy_at(s), cl.z_at(s) + 3.0])}
        if pose is not None and np.isfinite(c):
            pos, quat = np.array(pose[0]), np.array(pose[1])
            people = _people(space, pos, quat, num_carriers)
            if people_tris is None and people:
                people_tris = people[0][1].ravel().tolist()
            frame.update(clearance=float(c), box=flat(_box_corners(pos, quat)),
                         people=[flat(v) for v, _ in people])
        frames.append(frame)
    worst = min(range(len(frames)), key=lambda i: abs(frames[i]["s"] - b["s"]))
    return {"frames": frames, "box_tris": BOX_TRIS, "people_tris": people_tris or [],
            "bottleneck": {"frame": worst, "capacity_cm": float(b["capacity_cm"]),
                           "contact": flat(b["contact"]["xyz"]),
                           "part": b["contact"]["part"]},
            "furniture_only_cm": float(result["bottleneck_furniture_only"]["capacity_cm"]),
            "length": float(cl.s_total)}
