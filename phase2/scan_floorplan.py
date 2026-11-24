"""
スキャンから、階ごとの間取り図(上から見た図)を作る。

scan_picker の「出発点と目的地を床に置く」図に使う。以前は撮れた床の点を
高さで色分けして出していたが、壁も部屋の形も無く、階段のように床が上下に
重なる所は利用者が高さを数字で絞る必要があった。ここでは階を自動で分け、
階ごとに「床・階段・家具など・壁」のマス目の図にする。

判定(オラクル・掃引)には使わない。図を描くためだけの、粗い分類。

メッシュは Z が上・cm(scan_ingest.load_scan_zup の戻り値)。
"""

import numpy as np

# 階を見つけるときの値。実データ(家全体・階段だけのスキャン)で、床の面積を
# 高さ10cmごとに集計すると、階は数㎡の山になり、段は1段あたり0.15〜0.3㎡、
# 踊り場は1㎡弱だった。
LEVEL_BIN = 10.0          # 高さを集計する刻み(cm)
LEVEL_MIN_AREA = 1.0e4    # これ以上の面積(cm²)がある高さを階の候補にする
LEVEL_MERGE = 100.0       # この高さの差(cm)未満の山は同じ階(玄関の段差など)


def find_levels(z, area):
    """上向きの面の高さ z と面積 area から、階の床の高さ(低い順)を返す。

    面積の大きい高さを階の候補にし、近い候補は1つの階にまとめる。まとめた
    中で一番広い高さを、その階の床の高さにする(机やベッドの上面が床より
    狭ければ、床のほうが選ばれる)。小さなスキャンで候補が1つも無いときは、
    一番広い高さを階にする。

    「一番広い階に比べて狭い高さは階にしない」という条件は付けない。運び込みの
    経路だけを撮ると、2階は廊下だけで1階よりずっと狭くなり、2階が階に
    ならず置けなくなるため。
    """
    z = np.asarray(z, dtype=float)
    area = np.asarray(area, dtype=float)
    if len(z) == 0:
        return []
    lo = np.floor(z.min() / LEVEL_BIN) * LEVEL_BIN
    idx = ((z - lo) // LEVEL_BIN).astype(int)
    hist = np.bincount(idx, weights=area)
    big = np.flatnonzero(hist >= LEVEL_MIN_AREA)
    if len(big) == 0:
        big = np.array([int(np.argmax(hist))])
    groups = [[int(big[0])]]
    for b in big[1:]:
        if (b - groups[-1][-1]) * LEVEL_BIN < LEVEL_MERGE:
            groups[-1].append(int(b))
        else:
            groups.append([int(b)])
    levels = []
    for g in groups:
        b = max(g, key=lambda i: hist[i])
        m = idx == b
        levels.append(float(np.average(z[m], weights=area[m])))
    return levels


# ---- マス目の図 ----

CELL = 5.0            # マス目の1辺(cm)
FLOOR_BAND = 30.0     # 階の床の高さから上下この幅(cm)の上向きの面を、その階の床にする
STAIR_SPAN = 60.0     # 階の間のかたまりで、高さの幅がこれ以上(段が3つほど続く)なら階段
WALL_LO, WALL_HI = 30.0, 200.0   # 床からこの高さの範囲にある縦向きの面を壁の候補にする
WALL_MIN_SPAN = 60.0  # 1マスの中で縦にこれ以上続いていれば壁(段の蹴込みや低い家具は外す)
STAIR_TOUCH = 3       # 階段を出すのは、その階の床からこのマス数以内につながるものだけ
BLOCK_BELOW = 150.0   # 床の上にこの高さ(cm)未満の上向きの面(机など)があれば、そこには置けない
DENSITY = 0.3         # 面の上に撒く点の数(1cm²あたり)
MAX_POINTS = 600_000

# マスの種類
NOTHING, FLOOR, STAIR, OTHER, WALL = 0, 1, 2, 3, 4


class _Grid:
    """xy をマス目に割り当て、マスごとの個数・平均・最小・最大を出す。"""

    def __init__(self, bounds_xy, cell):
        (x0, y0), (x1, y1) = np.asarray(bounds_xy, dtype=float)
        self.cell = float(cell)
        self.x0 = float(np.floor(x0 / cell) * cell)
        self.y0 = float(np.floor(y0 / cell) * cell)
        self.nx = max(1, int(np.ceil((x1 - self.x0) / cell)))
        self.ny = max(1, int(np.ceil((y1 - self.y0) / cell)))

    def index(self, pts):
        ix = np.clip(((pts[:, 0] - self.x0) // self.cell).astype(int), 0, self.nx - 1)
        iy = np.clip(((pts[:, 1] - self.y0) // self.cell).astype(int), 0, self.ny - 1)
        return iy * self.nx + ix

    def stats(self, pts):
        """(点があるマス, 平均の高さ, 最小の高さ, 最大の高さ) を (ny, nx) で返す。"""
        n = self.nx * self.ny
        idx = self.index(pts)
        z = pts[:, 2]
        count = np.bincount(idx, minlength=n)
        with np.errstate(invalid="ignore"):
            mean = np.bincount(idx, weights=z, minlength=n) / count
        zmin = np.full(n, np.inf)
        zmax = np.full(n, -np.inf)
        np.minimum.at(zmin, idx, z)
        np.maximum.at(zmax, idx, z)
        shape = (self.ny, self.nx)
        return (count.reshape(shape) > 0, mean.reshape(shape),
                zmin.reshape(shape), zmax.reshape(shape))


def _stairs_between(grid, pts):
    """階と階の間にある上向きの点を、階段のかたまりとそれ以外のマスに分ける。

    上から見てつながったかたまりごとに高さの幅を見る。段が続いていれば幅が
    大きく、机やベッドの上面は小さい。踊り場は段とつながっているので階段に
    入る。(階段のかたまりのリスト, それ以外のマス, マスの高さ) を返す。
    """
    from scipy import ndimage

    has, mean, zmin, zmax = grid.stats(pts)
    # 段と段の間の撮りこぼしで切れないよう、1マス広げてからつながりを見る
    labels, n = ndimage.label(ndimage.binary_dilation(has), structure=np.ones((3, 3)))
    stairs = []
    if n:
        ids = np.arange(1, n + 1)
        where = np.where(has, labels, 0)
        lo = ndimage.minimum(zmin, where, ids)
        hi = ndimage.maximum(zmax, where, ids)
        stairs = [where == i for i in ids[(hi - lo) >= STAIR_SPAN]]
    other = has.copy()
    for comp in stairs:
        other &= ~comp
    return stairs, other, mean


def plans_from_points(up, side, levels, bounds_xy, cell=CELL):
    """上向きの面の点 up と縦向きの面の点 side から、階ごとの図を作る。

    戻り値は {"cell", "x0", "y0", "nx", "ny", "floors": [...]}。floors の
    各要素は {"name", "level", "kind", "z"} で、kind は (ny, nx) のマスの種類
    (NOTHING / FLOOR / STAIR / OTHER / WALL)、z は床と階段のマスの高さ(それ
    以外は nan)。行が y、列が x で、マス (iy, ix) の中心は
    (x0 + (ix + 0.5) * cell, y0 + (iy + 0.5) * cell)。

    階段は、床がつながっている階の図に出す。ふつうは上下の両方の階に出る
    (どちらの階から見ても置けるように)。片方の階にしかつながっていない
    かたまり(途中までしか撮れていない階段など)は、その階にだけ出す。
    一番上の階より上へ続くかたまりも同じ(上の階が狭くて階として見つから
    なかったときや、階段と踊り場だけを撮ったときに、階段へ置けるように)。
    """
    from scipy import ndimage

    up = np.asarray(up, dtype=float).reshape(-1, 3)
    side = np.asarray(side, dtype=float).reshape(-1, 3)
    grid = _Grid(bounds_xy, cell)
    shape = (grid.ny, grid.nx)
    empty = np.zeros(shape, dtype=bool)
    nothing = ([], empty, np.full(shape, np.nan))

    # 各階の床
    floor_cells = []
    for level in levels:
        m = np.abs(up[:, 2] - level) <= FLOOR_BAND
        has, mean, _, _ = grid.stats(up[m]) if m.any() else (empty, nothing[2], None, None)
        floor_cells.append((has, mean))

    # 階と階の間(と、一番下の階より下・一番上の階より上)にある上向きの面
    edges = [-np.inf] + list(levels) + [np.inf]
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (up[:, 2] > lo + FLOOR_BAND) & (up[:, 2] < hi - FLOOR_BAND)
        if not m.any():
            gaps.append(nothing)
        elif np.isfinite(lo):
            gaps.append(_stairs_between(grid, up[m]))
        else:
            has, mean, _, _ = grid.stats(up[m])   # 一番下の階より下。階段にはしない
            gaps.append(([], has, mean))

    floors = []
    for i, level in enumerate(levels):
        kind = np.full(shape, NOTHING, dtype=np.uint8)
        z = np.full(shape, np.nan)
        below, above = gaps[i], gaps[i + 1]
        # 家具など: この階の上にある、階段でない上向きの面。一番下の階には、
        # それより下の面も入れる
        kind[above[1] | (below[1] if i == 0 else empty)] = OTHER
        has, mean = floor_cells[i]
        near = ndimage.binary_dilation(has, iterations=STAIR_TOUCH)
        for stairs, _, stair_z in (below, above):
            for comp in stairs:
                if (comp & near).any():
                    kind[comp] = STAIR
                    z[comp] = stair_z[comp]
        # どの階の床にもつながらない階段のかたまりは、下の階に「家具など」で出す
        upper = floor_cells[i + 1][0] if i + 1 < len(levels) else empty
        near_upper = ndimage.binary_dilation(upper, iterations=STAIR_TOUCH)
        for comp in above[0]:
            if not (comp & near).any() and not (comp & near_upper).any():
                kind[comp] = OTHER
        # 床が撮れていても、すぐ上に机などの上面があるマスは「家具など」のままにする
        # (高い棚の上面や鴨居の下は歩けるので、床にする)
        blocked = above[1] & (above[2] < level + BLOCK_BELOW)
        free = has & ~blocked
        kind[free] = FLOOR
        z[free] = mean[free]
        m = (side[:, 2] >= level + WALL_LO) & (side[:, 2] <= level + WALL_HI)
        if m.any():
            w_has, _, zmin, zmax = grid.stats(side[m])
            kind[w_has & ((zmax - zmin) >= WALL_MIN_SPAN)] = WALL
        z[(kind != FLOOR) & (kind != STAIR)] = np.nan
        floors.append({"name": f"{i + 1}階", "level": float(level), "kind": kind, "z": z})
    return {"cell": grid.cell, "x0": grid.x0, "y0": grid.y0, "nx": grid.nx, "ny": grid.ny,
            "floors": floors}


def build_plans(mesh, cell=CELL, seed=0):
    """スキャンのメッシュから、階ごとの図(plans_from_points の戻り値)を作る。"""
    import trimesh

    import scan_space as SS

    nz = SS._floor_facing_z(mesh)
    area = mesh.area_faces
    zc = mesh.triangles_center[:, 2]
    up_faces = np.flatnonzero(nz > 0.9)       # scan_picker が床としていた面と同じ基準
    side_faces = np.flatnonzero(np.abs(nz) < 0.3)
    levels = find_levels(zc[up_faces], area[up_faces])

    def sample(faces):
        if len(faces) == 0:
            return np.zeros((0, 3))
        n = int(min(MAX_POINTS, max(1000, area[faces].sum() * DENSITY)))
        pts, _ = trimesh.sample.sample_surface(mesh.submesh([faces], append=True), n,
                                               seed=seed)
        return np.asarray(pts)

    return plans_from_points(sample(up_faces), sample(side_faces), levels,
                             mesh.bounds[:, :2], cell)


def to_json(plans):
    """図を、ブラウザに送れる形(リストと数値)にする。高さの無いマスは None。"""
    floors = []
    for f in plans["floors"]:
        z = np.round(f["z"], 1).astype(object)
        z[np.isnan(f["z"])] = None
        floors.append({"name": f["name"], "level": round(f["level"], 1),
                       "kind": f["kind"].tolist(), "z": z.tolist()})
    return {**{k: plans[k] for k in ("cell", "x0", "y0", "nx", "ny")}, "floors": floors}
