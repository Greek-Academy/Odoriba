"""
検証用の合成経路: 玄関 -> 廊下 -> 曲がり角 -> ドア(枠・開いた扉の板・ドアノブ)。

直方体で組むので「正解」(直方体のまま測った距離と床)が分かっている。
mesh_env.env_to_mesh でメッシュにして scan_space に渡し、スキャン経由の
判定が正解と一致するか、壁に穴を開けても判定が変わらないかを確かめる。

    python route_env.py              # 正解との突き合わせ + 穴あけテスト

寸法(cm)は一般的な戸建てを想定した仮値:
  - 玄関の土間から上がり框で 20cm 上がる
  - 廊下の幅 85cm、天井は廊下の床から 240cm
  - ドア枠の内寸 75cm(両側の枠 5cm)、鴨居は廊下の床から 200cm
  - 扉は 90 度開いて廊下の壁沿いにある(厚さ 3.5cm)。ドアノブは扉から 6cm 出る
"""

import argparse
import time

import numpy as np

# ---- 寸法(cm) ----
HALL_W = 85.0          # 廊下の幅
GENKAN_D = 100.0       # 土間の奥行き(y方向)
STEP_H = 20.0          # 上がり框の高さ
CEIL = 240.0           # 廊下の床から天井まで
LEG1 = 500.0           # 1本目の廊下(y方向)の長さ(土間を含む)
LEG2 = 500.0           # 2本目の廊下(x方向)の終点
DOOR_X = 300.0         # ドア枠の位置(2本目の廊下の途中)
JAMB_W = 5.0           # ドア枠の片側の幅(開口部が狭くなる分)
JAMB_T = 12.0          # ドア枠の厚み(x方向)
DOOR_H = 200.0         # 鴨居の高さ(廊下の床から)
LEAF_T = 3.5           # 扉の厚さ
LEAF_W = 72.0          # 扉の幅
KNOB = 6.0             # ドアノブの出っ張り
KNOB_FROM_EDGE = 6.0   # ドアノブの位置(扉の先端から)

Y2 = LEG1 - HALL_W     # 2本目の廊下の y の下端


def aabb(x0, x1, y0, y1, z0, z1):
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    h = np.array([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2])
    return (c, np.eye(3), h)


def build_route():
    """(outer, obstacles) を返す(phase2_lstair.build_lstairs と同じ形)。"""
    top = STEP_H + CEIL
    outer = [
        aabb(0, HALL_W, 0, LEG1, 0, top),     # 土間+1本目の廊下
        aabb(0, LEG2, Y2, LEG1, 0, top),      # 2本目の廊下
    ]
    f = STEP_H
    obstacles = [
        # 廊下の床(土間より上がり框の分高い)
        aabb(0, HALL_W, GENKAN_D, LEG1, 0, f),
        aabb(HALL_W, LEG2, Y2, LEG1, 0, f),
        # ドア枠(両側の縦枠と鴨居)
        aabb(DOOR_X, DOOR_X + JAMB_T, Y2, Y2 + JAMB_W, f, top),
        aabb(DOOR_X, DOOR_X + JAMB_T, LEG1 - JAMB_W, LEG1, f, top),
        aabb(DOOR_X, DOOR_X + JAMB_T, Y2, LEG1, f + DOOR_H, top),
        # 90度開いた扉: 吊り元(y=Y2+JAMB_W 側)から廊下の壁沿いに x 方向へ
        aabb(DOOR_X + JAMB_T, DOOR_X + JAMB_T + LEAF_W,
             Y2 + JAMB_W, Y2 + JAMB_W + LEAF_T, f + 1.0, f + DOOR_H - 1.0),
        # ドアノブ(扉の先端近く、廊下側へ出っ張る)
        aabb(DOOR_X + JAMB_T + LEAF_W - KNOB_FROM_EDGE - 6.0,
             DOOR_X + JAMB_T + LEAF_W - KNOB_FROM_EDGE,
             Y2 + JAMB_W + LEAF_T, Y2 + JAMB_W + LEAF_T + KNOB, f + 90.0, f + 96.0),
    ]
    return outer, obstacles


def floor_z(x, y, z_hint=None):
    """正解の床の高さ(土間は0、廊下は上がり框の高さ)。"""
    if 0.0 <= x <= HALL_W and 0.0 <= y < GENKAN_D:
        return 0.0
    if (0.0 <= x <= HALL_W and GENKAN_D <= y <= LEG1) or \
       (0.0 <= x <= LEG2 and Y2 <= y <= LEG1):
        return STEP_H
    return None


START_XYZ = (HALL_W / 2, 20.0, 0.0)            # 土間の奥
GOAL_XYZ = (LEG2 - 20.0, Y2 + HALL_W / 2, STEP_H)  # 2本目の廊下の突き当たり


class BoxSpace:
    """正解側: 中心線だけスキャン空間のものを借り、距離と床は直方体のまま測る。
    scan_plan の関数に ScanSpace の代わりに渡せる(同じ中心線で比べるため)。"""

    def __init__(self, outer, obstacles, space, floor_fn=floor_z):
        import phase2_demo as p
        self._clr = lambda pts: p.clearance_points_3d(pts, outer, obstacles)
        self._floor = floor_fn
        self.centerline = space.centerline
        self.lo, self.shape, self.h = space.lo, space.shape, space.h

    def clearance_points(self, pts):
        return self._clr(pts)

    def floor_fn(self, x, y, z_hint=None):
        return self._floor(x, y, z_hint)


def punch_holes(mesh, centers, radius, max_edge=4.0):
    """メッシュを細かく分割し、centers の各点から radius 以内の三角形を
    抜いて穴を開ける(撮れていない壁の部分の代わり)。"""
    import trimesh
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge)
    m = trimesh.Trimesh(v, f, process=False)
    tc = m.triangles_center
    keep = np.ones(len(f), dtype=bool)
    for c in centers:
        keep &= np.linalg.norm(tc - np.asarray(c), axis=1) > radius
    return trimesh.Trimesh(v, f[keep], process=False)


def main():
    import mesh_env as M
    import phase2_lstair as L
    import scan_plan as SP
    import scan_space as SS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carriers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    outer, obstacles = build_route()
    mesh = M.env_to_mesh(outer, obstacles)
    seed_xyz = (HALL_W / 2, 40.0, 100.0)

    def run(label, space, ref=None):
        t0 = time.time()
        out = {}
        for wh in (False, True):
            sw = SP.sweep_capacity(space, num_carriers=args.carriers, with_human=wh)
            fin = [r for r in sw if np.isfinite(r[1])]
            s, c, pose = min(fin, key=lambda r: r[1])
            xyz, cp, part = SP.worst_contact(space, pose, wh, args.carriers)
            out[wh] = (s, c, sw)
            print(f"  {label} {'2人で運ぶ' if wh else '家具だけ'}: 一番狭い所 s={s:.0f}cm "
                  f"xy={np.round(space.centerline.xy_at(s), 0).tolist()} 余裕{c:+.1f}cm "
                  f"/ 一番厳しい点 {np.round(xyz, 0).tolist()} ({part})")
        print(f"  [{time.time() - t0:.0f}s]")
        return out

    print(f"家具 {L.FURN_L:.0f}x{L.FURN_W:.0f}x{L.FURN_H:.0f}cm / 運搬者{args.carriers}人")
    t0 = time.time()
    space = SS.ScanSpace(mesh, seed_xyz)
    cl = space.build_centerline(START_XYZ, GOAL_XYZ)
    print(f"スキャン空間を構築 [{time.time() - t0:.0f}s]: 中心線 {cl.s_total:.0f}cm, "
          f"コーナー {[tuple(round(v) for v in c) for c in cl.corners]}, "
          f"ドア枠は s={cl.project(DOOR_X, Y2 + HALL_W / 2):.0f}cm")

    print("\n正解(直方体のまま、同じ中心線):")
    truth = run("正解", BoxSpace(outer, obstacles, space))
    print("\nスキャン経由:")
    scan = run("スキャン", space)

    # 壁に穴を開けても変わらないか(小さい穴は通れないとみなす LEAK_R の確認)
    holes = [(HALL_W, 250.0, 120.0),            # 1本目の廊下の右の壁
             (200.0, LEG1, 150.0),               # 2本目の廊下の奥の壁
             (DOOR_X + 150.0, Y2, 60.0)]         # ドアの先の壁
    for r in (6.0, 15.0):
        holed = punch_holes(mesh, holes, r)
        sp2 = SS.ScanSpace(holed, seed_xyz)
        sp2.centerline = space.centerline
        print(f"\n壁に半径{r:.0f}cmの穴を{len(holes)}つ開けた場合 "
              f"(穴から外へ漏れた警告: {'あり' if sp2.leaked else 'なし'}):")
        res = run(f"穴{r:.0f}cm", sp2)
        for wh in (False, True):
            d = max(abs(a[1] - b[1]) for a, b in zip(res[wh][2], scan[wh][2])
                    if np.isfinite(a[1]) and np.isfinite(b[1]))
            print(f"    穴なしとの差(掃引の各位置の最大): {d:.2f}cm "
                  f"({'2人' if wh else '家具だけ'})")

    for wh in (False, True):
        d = max(abs(a[1] - b[1]) for a, b in zip(truth[wh][2], scan[wh][2])
                if np.isfinite(a[1]) and np.isfinite(b[1]))
        print(f"正解との差(掃引の各位置の最大, {'2人' if wh else '家具だけ'}): {d:.2f}cm")


if __name__ == "__main__":
    main()
