"""
スキャン空間(scan_space.ScanSpace)の上での判定: 中心線に沿った誘導付きの
RRT探索と、ボトルネック掃引(どこで何cm足りないか)。

phase2_lstair の make_sampler / sweep_capacity は L字の中心線(skeleton_xy・
S_TOTAL・手で決めたコーナー区間)に依存している。ここではそれを
scan_space.Centerline(床の上を歩く経路から自動で作った中心線と、
進行方向が急に変わる所として自動で見つけたコーナー区間)で動く形にした。
可搬性オラクルは phase2_lstair のものをそのまま使う(ScanSpace を outer に
渡し、obstacles は None、床の高さは space.floor_fn)。

    python scan_plan.py <mesh.obj> --start X Y Z --goal X Y Z
"""

import numpy as np

import geometry3d as g3
import human_figure as hf
import phase2_demo as p
import phase2_lstair as L


def _yaw_range(cl, s0, s1):
    """コーナー区間 (s0, s1) の入口と出口の進行方向(度、小さい順)。"""
    a, b = cl.heading_at(s0), cl.heading_at(s1)
    return min(a, b), max(a, b)


def min_center_z(cl, s, pitch_deg):
    """弧長sで長軸をpitchだけ傾けたとき、家具の最下点が床に触れる中心高さ。
    phase2_lstair.min_center_z の中心線版。"""
    pr = np.radians(abs(pitch_deg))
    return cl.z_at(s) + (L.FURN_L / 2) * np.sin(pr) + (L.FURN_H / 2) * np.cos(pr)


def make_sampler(space, with_human, p_guided=0.8, pos_noise=10.0, rot_noise_deg=8.0):
    """中心線に沿った誘導付きサンプラー(phase2_lstair.make_sampler の汎用版)。

    - 位置: 中心線上の弧長sを一様に引き、横方向・高さにノイズ
    - 姿勢: 直線区間では長軸を進行方向に向け、ピッチは床の勾配。
      コーナー区間ではヨーを入口から出口の向きまで一様に、ピッチも一様
      (家具単体は立てて回す解を探せるように85度以上も撒く)
    - 確率(1 - p_guided)では空間全体から完全ランダム
    """
    cl = space.centerline
    max_pitch = L.MAX_TILT_DEG if with_human else 92.0
    lo = space.lo
    hi = space.lo + np.array(space.shape) * space.h

    def sampler(rng):
        if rng.random() >= p_guided:
            return rng.uniform(lo, hi), g3.random_quaternion(rng)
        s = rng.uniform(0.0, cl.s_total)
        corner = cl.corner_at(s)
        if corner is not None:
            y0, y1 = _yaw_range(cl, *corner)
            yaw = rng.uniform(y0 - 15.0, y1 + 15.0)
            if with_human:
                pitch = rng.uniform(0.0, max_pitch)
            else:
                pitch = rng.uniform(55.0, 92.0) if rng.random() < 0.6 \
                    else rng.uniform(0.0, 55.0)
        else:
            yaw = cl.heading_at(s) + rng.normal(0.0, rot_noise_deg)
            pitch = cl.slope_at(s) + rng.normal(0.0, rot_noise_deg)
        roll = rng.normal(0.0, rot_noise_deg * 0.5)
        quat = L._pose(yaw, pitch, roll).as_quat()
        x, y = cl.xy_at(s)
        z = min_center_z(cl, s, pitch) + abs(rng.normal(0.0, 12.0))
        pos = np.array([x, y, z]) + rng.normal(0.0, pos_noise, size=3) * np.array([1, 1, 0.3])
        return pos, quat

    return sampler


def validator(space):
    """rrt_connect に渡す判定関数(床の高さはスキャン空間のもの)。"""
    def check(pos, quat, with_human, outer, obstacles, num_carriers=None):
        return L.state_valid(pos, quat, with_human, outer, obstacles, num_carriers,
                             floor_fn=space.floor_fn)
    return check


def pose_clearance(space, pos, quat, with_human, num_carriers=None):
    """1姿勢の余裕(cm)。運搬者ありなら可搬性オラクル、なしなら家具だけ。"""
    if with_human:
        c, _ = L.carriable_clearance_lstair(pos, quat, space, None, num_carriers,
                                            floor_fn=space.floor_fn)
        return c
    return L.furniture_clearance(pos, quat, space, None)


def sweep_capacity(space, num_carriers=None, ds=15.0, with_human=True, max_pitch=None):
    """中心線上の各弧長sで、粗い姿勢グリッドの中で達成できる最良の余裕(cm)。
    [(s, capacity_cm, best_pose), ...] を返す。phase2_lstair.sweep_capacity の
    中心線版(姿勢グリッドの考え方は同じ)。"""
    cl = space.centerline
    if max_pitch is None:
        max_pitch = L.MAX_TILT_DEG if with_human else 90.0
    pitches = [t for t in (0.0, 15.0, 30.0, 45.0, 55.0, 70.0, 85.0, 90.0) if t <= max_pitch]
    laterals = (-10.0, 0.0, 10.0)
    z_extras = (0.0, 12.0, 30.0)
    # 端に近すぎるsは、家具+運搬者の列が出発点・目的地の先(スキャンした
    # 範囲の外かもしれない所)にはみ出すだけなので掃引しない
    s_margin = L.FURN_L / 2 + p.CARRY_ARM + p.HUMAN_R + 10.0
    results = []
    for s in np.arange(s_margin, cl.s_total - s_margin + 1e-9, ds):
        x0, y0 = cl.xy_at(s)
        heading = cl.heading_at(s)
        corner = cl.corner_at(s)
        if corner is not None:
            y_lo, y_hi = _yaw_range(cl, *corner)
            n = max(2, int(np.ceil((y_hi - y_lo) / 22.5)) + 1)
            yaws = tuple(np.linspace(y_lo, y_hi, n))
        else:
            yaws = (heading,)
        hr = np.radians(heading)
        side = np.array([-np.sin(hr), np.cos(hr)])   # 中心線に直交する向き
        best = -np.inf
        best_pose = None
        for yaw in yaws:
            for pitch in pitches:
                quat = L._pose(yaw, pitch).as_quat()
                z_base = min_center_z(cl, s, pitch)
                for lat in laterals:
                    x, y = np.array([x0, y0]) + side * lat
                    for dz in z_extras:
                        pos = np.array([x, y, z_base + dz])
                        c = pose_clearance(space, pos, quat, with_human, num_carriers)
                        if c > best:
                            best = c
                            best_pose = (pos.tolist(), quat.tolist())
        results.append((float(s), float(best), best_pose))
    return results


def worst_contact(space, pose, with_human, num_carriers=None):
    """その姿勢で一番めり込んでいる(余裕が一番小さい)点の座標と余裕。

    家具の表面点と、運搬者の体(人型の骨格点。余裕は半径を引いた値)を
    合わせて調べる。「どこで何に当たったか」を3D表示で示すために使う
    (手すりに当たっているなら、外せば通るかをユーザーが判断できる)。
    戻り値: (xyz, clearance_cm, "furniture" | "carrier")
    """
    pos, quat = np.asarray(pose[0]), np.asarray(pose[1])
    fp = L.furniture_points(pos, quat)
    fc = space.clearance_points(fp)
    i = int(np.argmin(fc))
    best = (fp[i], float(fc[i]), "furniture")
    if with_human and L.BODY_MODEL == "humanoid":
        _, off = L.carriable_clearance_lstair(pos, quat, space, None, num_carriers,
                                              floor_fn=space.floor_fn)
        placed = L.place_humans_lstair(pos, quat, 0.0 if off is None else off,
                                       num_carriers, space.floor_fn)
        if placed is not None:
            R = g3.rotmat_from_quat(quat)
            for sign, (center, _) in zip(L._carrier_signs(len(placed)), placed):
                floor, facing, hands = L._carrier_frame(pos, R, sign,
                                                        0.0 if off is None else off, center)
                for pts, r in hf.body_parts(center[:2], floor, facing, hands, L.LEG_CLEAR):
                    c = space.clearance_points(pts) - r
                    j = int(np.argmin(c))
                    if c[j] < best[1]:
                        best = (pts[j], float(c[j]), "carrier")
    return np.asarray(best[0]), best[1], best[2]


def best_effort_path(space, tree):
    """RRTが失敗したとき、start側ツリーで中心線上を一番先まで進めた
    ノードまでの経路(phase2_lstair.best_effort_path の中心線版)。"""
    cl = space.centerline
    progress = [cl.project(n.pos[0], n.pos[1]) for n in tree]
    i = int(np.argmax(progress))
    path = []
    while i is not None:
        n = tree[i]
        path.append((n.pos, n.quat))
        i = n.parent
    path.reverse()
    return path, float(progress[int(np.argmax(progress))])


def state_at(space, s, z_extra=0.0):
    """中心線上の弧長sに、長軸を進行方向に向けて家具を床に置いた状態。
    出発点・目的地の状態を作るのに使う。"""
    cl = space.centerline
    x, y = cl.xy_at(s)
    pos = np.array([x, y, cl.z_at(s) + L.FURN_H / 2 + z_extra])
    return pos, L._pose(cl.heading_at(s), 0.0).as_quat()


def judge(space, num_carriers=2, max_iter=4000, seed=0, end_margin=None, sweep=True,
          rrt=True, log=print):
    """出発点→目的地を、家具単体と運搬者ありの両方で判定する。

    出発点・目的地の状態は中心線の両端から end_margin だけ内側に置く
    (家具と運搬者の列が中心線の端の先にはみ出さないように)。
    戻り値は phase2_lstair の結果JSONと同じ形の dict。
    """
    cl = space.centerline
    if end_margin is None:
        end_margin = L.FURN_L / 2 + p.CARRY_ARM + p.HUMAN_R + 10.0
    start, goal = state_at(space, end_margin), state_at(space, cl.s_total - end_margin)
    check = validator(space)
    result = {"meta": {"centerline_length": cl.s_total,
                       "corners": [list(c) for c in cl.corners],
                       "furniture": {"L": L.FURN_L, "W": L.FURN_W, "H": L.FURN_H},
                       "carrier": {"num_carriers": num_carriers, **L.body_meta()},
                       "planner": {"max_iter": max_iter, "seed": seed, "w_rot": L.W_ROT}}}
    for key, with_human in ((("box_only", False), ("with_carriers", True)) if rrt else ()):
        path, tree = p.rrt_connect(
            start, goal, with_human=with_human, outer=space, obstacles=None,
            num_carriers=num_carriers, max_iter=max_iter, seed=seed, w_rot=L.W_ROT,
            sampler=make_sampler(space, with_human), validator=check, return_trees=True)
        if path is not None:
            result[key] = {"found": True, "path": L._path_to_json(path)}
        else:
            be, reached = best_effort_path(space, tree)
            result[key] = {"found": False, "reached_s": reached,
                           "best_effort_path": L._path_to_json(be)}
        log(f"  {key}: {'PASS' if path is not None else 'BLOCKED'}")
    if sweep:
        for key, with_human in (("bottleneck_furniture_only", False), ("bottleneck", True)):
            sw = sweep_capacity(space, num_carriers=num_carriers, with_human=with_human)
            finite = [r for r in sw if np.isfinite(r[1])]
            s_min, c_min, pose_min = min(finite, key=lambda r: r[1])
            xyz, c_pt, what = worst_contact(space, pose_min, with_human, num_carriers)
            result[key] = {"s": s_min, "capacity_cm": c_min,
                           "xy": cl.xy_at(s_min).tolist(), "pose": pose_min,
                           "contact": {"xyz": xyz.tolist(), "clearance_cm": c_pt,
                                       "part": what},
                           "profile": [[s, c] for s, c, _ in sw]}
            log(f"  {key}: s={s_min:.0f}cm 余裕{c_min:+.1f}cm "
                f"(一番厳しい点 {np.round(xyz, 0).tolist()} = {what})")
    return result
