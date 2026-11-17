"""スキャンから、経路に沿った「壁から壁の幅」と「天井までの高さ」を測る。

搬入の下見でプロが見る量に合わせた計測。家具店の採寸シートは、回り階段では
扇形の段ごとに「幅」と「高さ」を測らせている(decisions.md 2026-11-18)。
measure_stairs の幅は「踏み面として撮れた面の広がり」で、壁から壁ではなく、
メジャーと7〜17cm合わなかった。ここでは壁そのものまでの距離を測る。

測り方:
  - メッシュの面の上に点をまんべんなく撒く(1cm²あたり数点)
  - 経路の各点で、床から決まった高さの水平な輪切りをとり、その点を通る
    弦(一方の壁から反対の壁まで)を向きを変えながら測って、一番短いものを
    その高さの幅とする。向きを決め打ちしないので、扇形の段でも測れる
  - 高さごとの幅の中央値 = 壁から壁(手すりは1〜2の高さにしか出ないので
    中央値には効かない。1つの高さだけで読むと、スキャンの荒れで数cm振れた)。
    全部の高さで一番狭い幅 = 手すりなどの出っ張りを含めた、家具が使える幅
  - 天井 = 真上にある下向きの面のうち一番低いものまでの高さ

メッシュは Z が上・cm(scan_ingest.load_scan_zup の戻り値)。
"""

import numpy as np
import trimesh

HEIGHTS = np.arange(40.0, 180.0, 10.0)  # 床から測る高さ(cm)
SLAB = 2.5         # 輪切りの厚みの半分(cm)
BEAM = 2.5         # 弦とみなす帯の幅の半分(cm)
REACH = 250.0      # これより遠い面は見ない(cm)
ANGLES = np.deg2rad(np.arange(0.0, 180.0, 3.0))


class _Cloud:
    """メッシュの面の上に撒いた点と、その面の向き。"""

    def __init__(self, mesh, density=5.0, max_pts=4_000_000, seed=0):
        n = int(min(max_pts, max(200_000, mesh.area * density)))
        pts, fi = trimesh.sample.sample_surface(mesh, n, seed=seed)
        self.pts = np.asarray(pts)
        nz = np.asarray(mesh.face_normals)[fi][:, 2]
        self.up = self.pts[nz > 0.8]      # 床・踏み面
        self.down = self.pts[nz < -0.5]   # 天井・上の階段の裏
        self._ca, self._sa = np.cos(ANGLES), np.sin(ANGLES)

    def floor_z(self, x, y, z_hint, r=6.0, dz=40.0):
        """(x, y) の真下の床の高さ。z_hint の近くの上向きの面から読む。"""
        c = self.up[np.hypot(self.up[:, 0] - x, self.up[:, 1] - y) < r]
        c = c[np.abs(c[:, 2] - z_hint) < dz]
        return float(np.median(c[:, 2])) if len(c) >= 5 else None

    def ceiling(self, x, y, fz, r=8.0, min_gap=100.0):
        """床 fz から真上の天井までの高さ。真上に面が無ければ nan。"""
        c = self.down[(np.hypot(self.down[:, 0] - x, self.down[:, 1] - y) < r)
                      & (self.down[:, 2] > fz + min_gap)]
        return float(c[:, 2].min() - fz) if len(c) else float("nan")

    def min_chord(self, x, y, z):
        """高さ z の輪切りで、(x, y) を通る一番短い弦の長さと、その向き(度)。

        弦の両側に面が無い(スキャンの穴・開いた空間)向きは無限大として扱う。
        """
        s = self.pts[np.abs(self.pts[:, 2] - z) < SLAB]
        d = s[:, :2] - (x, y)
        d = d[np.hypot(d[:, 0], d[:, 1]) < REACH]
        along = d[:, 0:1] * self._ca + d[:, 1:2] * self._sa
        perp = -d[:, 0:1] * self._sa + d[:, 1:2] * self._ca
        ok = np.abs(perp) < BEAM
        fwd = np.where(ok & (along > 3.0), along, np.inf).min(axis=0, initial=np.inf)
        bwd = np.where(ok & (along < -3.0), -along, np.inf).min(axis=0, initial=np.inf)
        c = fwd + bwd
        k = int(np.argmin(c))
        return float(c[k]), float(np.rad2deg(ANGLES[k]))


def section_profile(mesh, path, heights=HEIGHTS, density=5.0):
    """経路 path((N,3)、cm)の各点で、幅と天井を測る。

    戻り値: 点ごとの dict のリスト。キー:
      s(経路に沿った距離), x, y, floor_z,
      width_wall(高さごとの幅の中央値 = 壁から壁),
      width_min(全部の高さで一番狭い幅), width_min_h(その高さ), ceiling
    床が読めない点(スキャンの穴)は飛ばす。面が無くて測れない幅は nan。
    """
    cloud = _Cloud(mesh, density=density)
    path = np.asarray(path, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    out = []
    for si, (x, y, z) in zip(s, path):
        fz = cloud.floor_z(x, y, z)
        if fz is None:
            continue
        ws = np.array([cloud.min_chord(x, y, fz + h)[0] for h in heights])
        k = int(np.argmin(ws))
        fin = ws[np.isfinite(ws)]
        out.append({
            "s": float(si), "x": float(x), "y": float(y), "floor_z": fz,
            "width_wall": float(np.median(fin)) if len(fin) else float("nan"),
            "width_min": float(ws[k]) if np.isfinite(ws[k]) else float("nan"),
            "width_min_h": float(heights[k]),
            "ceiling": cloud.ceiling(x, y, fz),
        })
    return out


def by_step(profile, gap=8.0):
    """経路の点を、床の高さが同じもの同士(=同じ段・同じ床)にまとめる。

    床の高さが gap(cm) より大きく変わったら次の段とみなす。
    戻り値: 段ごとの dict のリスト。キー:
      index(0 が出発点の床), floor_z, n(点の数),
      width_wall(壁から壁の中央値), width_wall_min(同じく最小),
      width_min(出っ張り込みの一番狭い幅), ceiling(段の中で一番低い天井)
    """
    groups = []
    for p in profile:
        if groups and abs(p["floor_z"] - groups[-1][-1]["floor_z"]) <= gap:
            groups[-1].append(p)
        else:
            groups.append([p])

    def stat(g, key, fn):
        a = np.array([p[key] for p in g])
        a = a[np.isfinite(a)]
        return float(fn(a)) if len(a) else float("nan")

    return [{
        "index": i, "floor_z": stat(g, "floor_z", np.median), "n": len(g),
        "width_wall": stat(g, "width_wall", np.median),
        "width_wall_min": stat(g, "width_wall", np.min),
        "width_min": stat(g, "width_min", np.min),
        "ceiling": stat(g, "ceiling", np.min),
    } for i, g in enumerate(groups)]


if __name__ == "__main__":
    import argparse

    import scan_ingest as si
    import scan_space as ss

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="スキャン OBJ/PLY/zip")
    ap.add_argument("--assume-units", choices=("m", "cm", "mm"))
    ap.add_argument("--seed", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="自由空間の中の1点(cm)。省くと自動")
    ap.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "Z"))
    args = ap.parse_args()

    mesh, _ = si.load_scan_zup(args.scan, args.assume_units)
    space = ss.ScanSpace(mesh, tuple(args.seed) if args.seed else ss.auto_seed(mesh))
    if args.start and args.goal:
        start, goal = args.start, args.goal
    else:
        start, goal, _ = space.propose_endpoints()
    cl = space.build_centerline(start, goal)
    path = np.array([[*cl.xy_at(s), cl.z_at(s)] for s in np.arange(0.0, cl.s_total, 5.0)])
    steps = by_step(section_profile(mesh, path))
    print("経路に沿った幅と天井(0 が出発点の床。単位 cm):")
    print("  番号  床の高さ  壁から壁(中央/最小)  出っ張り込みの最小  天井")
    for st in steps:
        print(f"  {st['index']:3d}  {st['floor_z']:7.0f}  "
              f"{st['width_wall']:7.1f} /{st['width_wall_min']:6.1f}  "
              f"{st['width_min']:12.1f}  {st['ceiling']:8.0f}")
