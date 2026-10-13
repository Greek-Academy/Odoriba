"""
スキャンした空間をそのまま判定に使う: メッシュ1つから、可搬性オラクルと
プランナーが使う「距離場・床の高さ・中心線」を作る。

方針は「撮ったままを判定する」。手すり・扉・ドアノブを意味として認識
しない(どこまでが手すりかの判定や、取り外しの考慮はしない)。外すもの
(扉・外せる手すり)は、ユーザーが外した・開けた状態で撮る前提にする。
判定結果は「どこで何cm足りないか」と、そこで何に当たったかの座標で返し、
外せば通るかはユーザーが判断する。

型として用意した階段(phase2_lstair など)と違い、玄関から部屋までの
経路全体(廊下・曲がり角・ドア・階段)を同じ仕組みで扱う:

  1. 距離場  -- メッシュ表面に点を細かく撒いて格子(既定2cm)に割り当て、
     距離変換で各格子点から一番近い面までの距離を求める。内側/外側は
     水密化の代わりに、出発点から面に触れずにたどり着ける領域を
     塗りつぶして決める(スキャンは壁が開いていて水密化できないため)。
     半径 leak_r 未満の穴は通れないとみなし、撮れていない小さな穴から
     塗りつぶしが外へ漏れないようにする
  2. 床の高さ -- 自由空間の下側の境界のうち、人が立てる広さと頭上の
     余裕があるものを床とする。同じ(x, y)に複数の床(階段の上下など)が
     重なるので、高さのヒント(把持点の高さ)より下で一番高い床を返す
  3. 中心線  -- 床の上を出発点から目的地まで歩く経路を、壁から離れる
     ほど安いコストで探し、なめらかにして弧長sの関数にする

壁・天井の撮れていない部分が大きいスキャン(穴が半径 leak_r を超える)では、
塗りつぶしが外へ漏れてしまう。confine_to_floor=True(既定)では、撮れた床
(上向きの面)の真上 FLOOR_SPACE_H までだけを空間とみなし、その範囲の境目を
壁として扱う。撮れていない床の上(吹き抜けや、撮れなかった2階の床など)は
空間に含まれないので、そこを通る判定はできない(撮り直しが要る)。

ScanSpace は phase2_lstair のオラクル(carriable_clearance_lstair など)に
`outer` の代わりとして渡せる(obstacles は None)。オラクルは
clearance_points を持つ環境ならそれで距離を測る(env_clearance_points)。
"""

import numpy as np
import scipy.ndimage as nd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import trimesh

# ---- 既定のパラメータ(cm) ----
VOXEL = 2.0            # 距離場の格子の間隔
SAMPLE_SPACING = 0.5   # 表面に撒く点のおおよその間隔(格子より十分細かく)
LEAK_R = 8.0           # この半径未満の穴・隙間は通れないとみなす
HEADROOM = 150.0       # 床とみなすのに必要な頭上の余裕
MAX_STEP = 25.0        # 中心線で1歩に越えられる段差
WALK_CLEAR = 18.0      # 中心線を引く床の、腰の高さでの壁までの最小余裕
FLOOR_TOL = 5.0        # 高さのヒントより少し上の床まで許す(ヒントの誤差分)
ENDPOINT_PULL = 60.0   # 提案した出発点・目的地を、この範囲で通路の中央へ寄せる
BRIDGE_GAP = 30.0      # 床の撮りこぼし(ドアの敷居など)をこの幅まで橋渡しする
# 撮れた床の範囲で空間を区切る(confine_to_floor)ときの寸法
# 床からこの高さまでを空間とみなす(天井が撮れていない所の上限)。家具を傾けて
# 持ち上げると上端は床から約3mに達し、階段の吹き抜けは天井が高いので、
# 普通の天井(約2.4m)より十分高くしておく。天井が撮れていればそちらが効く
FLOOR_SPACE_H = 350.0
FLOOR_CLOSE = 10.0     # この幅以下の床の撮りこぼしは埋める
FLOOR_PAD = 6.0        # 床の範囲をこれだけ外へ広げる(実際の壁がある所では壁が効くように)


def _floor_facing_z(mesh):
    """各面の法線のz成分を「床なら正」になる向きにそろえて返す。

    スキャン(Scaniverseなど)の面は撮影者の側=自由空間の側を向くので、床の
    法線は上向き。一方、直方体からブーリアンで作った合成メッシュは自由空間の
    外側を向くので、床の法線は下向き(上向きは天井)になる。一番低い高さ帯
    (床があるはずの所)で上向き・下向きの水平な面の面積を比べ、下向きが
    多ければ向きを反転して扱う。
    """
    nz = mesh.face_normals[:, 2]
    zc = mesh.triangles_center[:, 2]
    z0, z1 = mesh.bounds[0][2], mesh.bounds[1][2]
    low = zc < z0 + 0.1 * (z1 - z0)
    area = mesh.area_faces
    up = area[low & (nz > 0.9)].sum()
    down = area[low & (nz < -0.9)].sum()
    return -nz if down > up else nz


def auto_seed(mesh, height=100.0, n=300, seed=0):
    """塗りつぶしの起点(自由空間の中の1点)を自動で選ぶ。

    一番低い階の床(上向きの面のうち、面積で見て下から5%の高さの±30cm)の
    上 height の点を n 個とり、メッシュの面から一番離れたものを返す。部屋や
    廊下の中央あたりになる。一番低い階にするのは、玄関は普通そこにあり、
    吹き抜けのような天井の高い所を選んで小さな床のかたまりから始めないため。
    """
    nz = _floor_facing_z(mesh)
    faces = np.flatnonzero(nz > 0.9)
    if len(faces) == 0:
        raise ValueError("床(上向きの面)が見つからない")
    fz = mesh.triangles_center[faces, 2]
    order = np.argsort(fz)
    cum = np.cumsum(mesh.area_faces[faces][order])
    z_low = fz[order][np.searchsorted(cum, 0.05 * cum[-1])]
    faces = faces[np.abs(fz - z_low) <= 30.0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(faces, size=min(n, len(faces)), replace=False,
                      p=mesh.area_faces[faces] / mesh.area_faces[faces].sum())
    cand = mesh.triangles_center[pick] + np.array([0.0, 0.0, height])
    # 小さな物の上面(ドアノブ・扉の上端など)から上へ伸ばすと天井より上
    # (スキャンの外)に出て、「面から遠い点」として選ばれてしまう。メッシュの
    # 上端より下で、真上に面(天井)がある候補に絞る。天井が撮れていない
    # スキャンで1つも残らなければ、天井の条件は外す。
    cand = cand[cand[:, 2] < mesh.bounds[1][2] - 20.0]
    if len(cand) == 0:
        raise ValueError("起点の候補が見つからない(床の上に空間がない)")
    covered = mesh.ray.intersects_any(cand, np.tile([0.0, 0.0, 1.0], (len(cand), 1)))
    if covered.any():
        cand = cand[covered]
    _, dist, _ = trimesh.proximity.closest_point(mesh, cand)
    return tuple(float(v) for v in cand[int(np.argmax(dist))])


class ScanSpace:
    """スキャンメッシュから作った距離場・床・中心線。

    mesh: trimesh.Trimesh(cm単位・Z-up。scan_ingest で整えたもの)
    start_xyz: 自由空間の中の1点(出発点の床から少し上など)。塗りつぶしの起点
    """

    def __init__(self, mesh, start_xyz, voxel=VOXEL, leak_r=LEAK_R,
                 sample_spacing=SAMPLE_SPACING, seed=0, confine_to_floor=True):
        self.h = float(voxel)
        self.leak_r = float(leak_r)
        h = self.h
        self.lo = mesh.bounds[0] - 3 * h
        hi = mesh.bounds[1] + 3 * h
        self.shape = tuple(int(n) for n in np.ceil((hi - self.lo) / h).astype(int) + 1)

        # ---- 1. 表面の点を格子に割り当てる ----
        n_samples = int(mesh.area / sample_spacing ** 2)
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
        vox = np.floor((pts - self.lo) / h + 0.5).astype(np.int64)
        lin = np.ravel_multi_index(vox.T, self.shape)
        # 各表面ボクセルの代表点 = 中心に一番近いサンプル点
        center = self.lo + vox * h
        off = np.linalg.norm(pts - center, axis=1)
        order = np.lexsort((off, lin))
        lin_sorted = lin[order]
        first = np.concatenate([[True], lin_sorted[1:] != lin_sorted[:-1]])
        surf_lin = lin_sorted[first]
        rep = pts[order][first].astype(np.float32)
        surf = np.zeros(self.shape, dtype=bool)
        surf.flat[surf_lin] = True

        # ---- 1b. 撮れた床の範囲で空間を区切る ----
        # 範囲の境目のボクセルを「面」に加える(代表点はボクセルの中心)。
        # 塗りつぶしは境目を壁と同じに扱う。距離は塗りつぶしの後で、自由空間に
        # 接する境目だけを残して計算し直す(床の下・壁の裏にある境目まで面に
        # すると、段や壁へのめり込みの深さを浅く見積もってしまうため)。
        self.allowed = None
        real_lin, real_rep = surf_lin, rep
        border = None
        if confine_to_floor:
            up = _floor_facing_z(mesh)[face_idx] > 0.9
            allowed = self._floor_space(vox[up])
            border = allowed & ~nd.binary_erosion(allowed) & ~surf
            add = np.flatnonzero(border)
            surf_lin = np.concatenate([surf_lin, add])
            add_c = self.lo + np.column_stack(np.unravel_index(add, self.shape)) * h
            rep = np.concatenate([rep, add_c.astype(np.float32)])
            surf |= border
            self.allowed = allowed

        # ---- 2. 符号なし距離 ----
        unsigned, ind, rep_of = self._unsigned_field(surf, surf_lin, rep)

        # ---- 3. 内側/外側: 出発点から塗りつぶす ----
        # (a) 壁から leak_r より離れた所だけをたどって出発点とつながる領域
        #     (中心部)を求める。半径 leak_r 未満の穴はここを通れない
        # (b) 中心部から leak_r + 格子1つ分以内で、表面ボクセルを横切らずに
        #     出発点とつながる所を候補にする
        # (c) 候補のうち「一番近い面から離れる向きに進むと中心部に近づく」
        #     所だけを自由空間とする。壁の手前なら面から離れると部屋の中央
        #     (中心部)へ向かうが、小さな穴で壁の裏とつながった所では面から
        #     離れると中心部から遠ざかる。距離の幅で絞るより壁際を取りこぼさない
        #     表面ボクセル自身も、中心が面の手前(自由空間側)にあればこの判定で
        #     自由空間に入れる(面が格子点の中間にあると、面を含むボクセルの
        #     中心は面から最大1cm手前にずれるため。入れないと壁の手前1〜2cmが
        #     負になる)
        start_v = self._voxel_index(start_xyz)
        if unsigned[start_v] <= leak_r:
            raise ValueError(f"出発点 {start_xyz} が壁から{leak_r:.0f}cm以内にある"
                             f"(余裕{unsigned[start_v]:.1f}cm)。もっと空間の中央に置く")
        core_lab, _ = nd.label(unsigned > leak_r)
        core = core_lab == core_lab[start_v]
        del core_lab
        d_core = nd.distance_transform_edt(~core, sampling=h).astype(np.float32)
        open_lab, _ = nd.label(~surf)
        cand = (d_core <= leak_r + h) & ((open_lab == open_lab[start_v]) | surf) & ~core
        del open_lab
        ci = np.nonzero(cand)
        v = self.lo + np.column_stack(ci) * h
        q = rep[rep_of[np.ravel_multi_index(tuple(ind[k][ci] for k in range(3)),
                                            self.shape)]]
        del ind
        away = v - q
        norm = np.linalg.norm(away, axis=1, keepdims=True)
        away = away / np.maximum(norm, 1e-6)
        probe = np.floor((v + away * 1.5 * h - self.lo) / h + 0.5).astype(np.int64)
        probe = np.clip(probe, 0, np.array(self.shape) - 1)
        ok = (d_core[tuple(probe.T)] < d_core[ci]) & (norm[:, 0] > 1e-6)
        free = core.copy()
        free[tuple(c[ok] for c in ci)] = True
        del d_core, cand
        # 自由空間がスキャン範囲(格子)の端まで届いた = 半径 leak_r 以上の穴から
        # 外へ漏れた可能性が高い。撮れていない壁があるので撮り直しを促す
        self.leaked = bool(free[0].any() or free[-1].any() or free[:, 0].any()
                           or free[:, -1].any() or free[:, :, 0].any()
                           or free[:, :, -1].any())
        if self.leaked:
            import warnings
            warnings.warn(f"自由空間がスキャン範囲の端まで届いた: 半径{leak_r:.0f}cm以上の"
                          "穴(撮れていない壁)がある可能性が高い。判定は信用できない")
        if border is not None:
            # 自由空間に接する境目だけを面として残し、距離を計算し直す
            keep = border & nd.binary_dilation(free)
            k_lin = np.flatnonzero(keep)
            surf2 = np.zeros(self.shape, dtype=bool)
            surf2.flat[real_lin] = True
            surf2.flat[k_lin] = True
            k_c = self.lo + np.column_stack(np.unravel_index(k_lin, self.shape)) * h
            unsigned, _, _ = self._unsigned_field(
                surf2, np.concatenate([real_lin, k_lin]),
                np.concatenate([real_rep, k_c.astype(np.float32)]), with_index=False)
        self.free = free
        self.phi = np.where(free, unsigned, -unsigned).astype(np.float32)

        self._build_floors()
        self.centerline = None

    def _unsigned_field(self, surf, surf_lin, rep, with_index=True):
        """各格子点から一番近い面までの距離(符号なし)。

        距離変換が返すのは「一番近い表面ボクセル」まで。そのボクセルの
        代表点(実際の面上の点)までの距離に置き換えて、格子の粗さによる
        誤差を小さくする。メモリを抑えるため x のスライスごとに処理する。
        戻り値: (unsigned, 一番近い表面ボクセルの添字 or None, 代表点の表引き)
        """
        h = self.h
        _, ind = nd.distance_transform_edt(~surf, return_indices=True)
        rep_of = np.full(int(np.prod(self.shape)), -1, dtype=np.int32)
        rep_of[surf_lin] = np.arange(len(surf_lin), dtype=np.int32)
        unsigned = np.empty(self.shape, dtype=np.float32)
        yy, zz = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[2]), indexing="ij")
        for i in range(self.shape[0]):
            near = np.ravel_multi_index((ind[0, i], ind[1, i], ind[2, i]), self.shape)
            q = rep[rep_of[near]]
            c = np.stack([self.lo[0] + i * h + 0 * yy, self.lo[1] + yy * h,
                          self.lo[2] + zz * h], axis=-1)
            unsigned[i] = np.linalg.norm(c - q, axis=-1)
        if not with_index:
            return unsigned, None, rep_of
        return unsigned, ind, rep_of

    def _floor_space(self, floor_vox):
        """上向きの面のボクセルから、空間とみなす範囲(3Dの真偽値)を作る。

        床のボクセルを水平方向に FLOOR_CLOSE + FLOOR_PAD 広げてから FLOOR_CLOSE
        縮め(撮りこぼしを埋めて FLOOR_PAD だけ外へ広げる)、各床から上へ
        FLOOR_SPACE_H、下へ格子2つ分を範囲にする。床が上下に重なる所
        (階段の下と上など)は、それぞれの床からの範囲の和になる。
        """
        h = self.h
        fv = np.zeros(self.shape, dtype=bool)
        fv[tuple(np.asarray(floor_vox).T)] = True
        xy = (h, h, 1e6)   # z方向の距離を極端に大きくして、水平方向だけで広げる
        grown = nd.distance_transform_edt(~fv, sampling=xy) <= FLOOR_CLOSE + FLOOR_PAD
        fv = nd.distance_transform_edt(grown, sampling=xy) > FLOOR_CLOSE
        del grown
        n_up = int(np.ceil(FLOOR_SPACE_H / h))
        n_down = 2
        K = self.shape[2]
        c = np.zeros(self.shape[:2] + (K + 1,), dtype=np.int32)
        np.cumsum(fv, axis=2, out=c[:, :, 1:])
        k = np.arange(K)
        # 高さ k が範囲内 <=> ある床の高さ k' について k' - n_down <= k <= k' + n_up
        # <=> 区間 [k - n_up, k + n_down] に床がある(累積和で数える)
        hi_i = np.minimum(k + n_down + 1, K)
        lo_i = np.maximum(k - n_up, 0)
        return (c[:, :, hi_i] - c[:, :, lo_i]) > 0

    # ---- 距離場の問い合わせ ----

    def _voxel_index(self, xyz):
        return tuple(int(v) for v in np.floor((np.asarray(xyz) - self.lo) / self.h + 0.5))

    def clearance_points(self, pts):
        """各点の符号付き余裕(cm)。自由空間の内側なら正、壁の中・
        スキャン範囲の外なら負。格子からの三線形補間。"""
        # 格子の外の点は、格子の端の値から「はみ出した距離」を引く
        # (一律の定数にすると、どれだけはみ出したかの比較ができなくなる)
        g = (np.asarray(pts, dtype=float) - self.lo) / self.h
        top = np.array(self.shape, dtype=float) - 1.0
        gc = np.clip(g, 0.0, top)
        out = np.linalg.norm(g - gc, axis=-1) * self.h
        return nd.map_coordinates(self.phi, gc.T, order=1, mode="nearest") - out

    # ---- 床 ----

    def _build_floors(self):
        """自由空間の下側の境界から床を求め、列(x, y)ごとの床の高さを持つ。

        床とみなす条件:
          - 下のボクセルが自由空間でない(下側の境界)
          - 頭上 HEADROOM 分の自由空間が続く(手すりの上面などを除く)
          - 周り1格子(±voxel)も同じ高さの床(人が立てる広さ。細い物の
            上面を除く)
        床の高さは「境界のボクセルの中心 - そこから面までの距離」。
        """
        h = self.h
        free = self.free
        below_solid = np.zeros_like(free)
        below_solid[:, :, 1:] = free[:, :, 1:] & ~free[:, :, :-1]
        # 頭上の余裕: 上向きに連続する自由空間の長さ(格子数)
        n_head = int(np.ceil(HEADROOM / h))
        run = np.zeros(free.shape, dtype=np.int32)
        for k in range(free.shape[2] - 1, -1, -1):
            nxt = run[:, :, k + 1] if k + 1 < free.shape[2] else 0
            run[:, :, k] = np.where(free[:, :, k], nxt + 1, 0)
        cand = below_solid & (run >= n_head)
        del run
        ci, cj, ck = np.nonzero(cand)
        z = self.lo[2] + ck * h - self.phi[ci, cj, ck]
        # 周り1格子に同じ高さ(±3cm)の床があるか
        key = {}
        for a, b, zz in zip(ci, cj, z):
            key.setdefault((int(a), int(b)), []).append(float(zz))
        floors = {}
        for (a, b), zs in key.items():
            ok = []
            for zz in zs:
                if all(any(abs(z2 - zz) <= 3.0 for z2 in key.get((a + da, b + db), ()))
                       for da in (-1, 0, 1) for db in (-1, 0, 1)):
                    ok.append(zz)
            if ok:
                floors[(a, b)] = sorted(ok)
        self.floors = floors

    def floor_fn(self, x, y, z_hint=None):
        """(x, y) で、高さ z_hint より下で一番高い床の高さ。床がなければNone。

        z_hint が None なら一番低い床を返す。運搬者オラクルは把持点の
        高さをヒントに渡す(階段の上下で床が重なる所で、正しい段を選ぶため)。
        """
        a = int(np.floor((x - self.lo[0]) / self.h + 0.5))
        b = int(np.floor((y - self.lo[1]) / self.h + 0.5))
        zs = self.floors.get((a, b))
        if not zs:
            return None
        if z_hint is None:
            return zs[0]
        below = [z for z in zs if z <= z_hint + FLOOR_TOL]
        return below[-1] if below else None

    # ---- 中心線 ----

    def _walk_graph(self):
        """床の上を歩くグラフ。戻り値: (各ノードの座標, 中央寄りを好むコストの
        グラフ, 長さだけのグラフ)。

        ノードは床の格子のうち腰の高さで壁から WALK_CLEAR 以上離れた所。
        隣接は段差 MAX_STEP 以下。中心線用のコストは壁から離れるほど安く
        (通路の中央寄りを通るように)、出発点・目的地の提案には長さだけを使う。
        """
        if getattr(self, "_graph_cache", None) is not None:
            return self._graph_cache
        h = self.h
        nodes = [(a, b, z) for (a, b), zs in self.floors.items() for z in zs]
        xyz = np.array([(self.lo[0] + a * h, self.lo[1] + b * h, z) for a, b, z in nodes])
        clr = self.clearance_points(xyz + np.array([0.0, 0.0, 100.0]))
        keep = clr >= WALK_CLEAR
        nodes = [n for n, k in zip(nodes, keep) if k]
        xyz, clr = xyz[keep], clr[keep]
        index = {}
        for n, (a, b, z) in enumerate(nodes):
            index.setdefault((a, b), []).append((z, n))
        # 隣接は3格子先まで: 段の縁の近くは床とみなさない列(周り1格子に
        # 同じ高さの床が要る)が段ごとに2列できるので、すぐ隣だけだと
        # 段の上り下りでつながらない
        reach = range(-3, 4)
        rows, cols, w, plain = [], [], [], []
        for n, (a, b, z) in enumerate(nodes):
            for da in reach:
                for db in reach:
                    if da == 0 and db == 0:
                        continue
                    for z2, m in index.get((a + da, b + db), ()):
                        dz = abs(z2 - z)
                        if dz <= MAX_STEP:
                            length = np.sqrt((da * h) ** 2 + (db * h) ** 2 + dz ** 2)
                            c = 0.5 * (clr[n] + clr[m])
                            rows.append(n)
                            cols.append(m)
                            w.append(length * (1.0 + (40.0 / c) ** 2))
                            plain.append(length)
        shape = (len(nodes), len(nodes))
        self._bridge(xyz, clr, rows, cols, w, plain)
        graph = coo_matrix((w, (rows, cols)), shape=shape).tocsr()
        lengths = coo_matrix((plain, (rows, cols)), shape=shape).tocsr()
        self._graph_cache = (xyz, graph, lengths)
        self._graph_clr = clr
        return self._graph_cache

    def _bridge(self, xyz, clr, rows, cols, w, plain, max_pairs=40):
        """床の撮りこぼし(ドアの敷居など)で途切れた床のかたまりどうしを
        橋渡しする辺を rows/cols/w/plain に足す。

        別のかたまりの床どうしで、水平に BRIDGE_GAP 以内・段差 MAX_STEP 以内、
        かつ間の腰の高さ(床から100cm)が WALK_CLEAR 以上空いている所だけを
        つなぐ。壁を挟んだ所は間の余裕が負になるのでつながらない。
        かたまりの組ごとに近い順に max_pairs 組まで調べる。
        """
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
        n = len(xyz)
        if n < 2:
            return
        base = coo_matrix((plain, (rows, cols)), shape=(n, n)).tocsr()
        n_comp, lab = connected_components(base, directed=False)
        if n_comp < 2:
            return
        sizes = np.bincount(lab)
        comps = [c for c in np.argsort(-sizes) if sizes[c] >= 20]
        trees = {c: (np.flatnonzero(lab == c),) for c in comps}
        for c in comps:
            idx = trees[c][0]
            trees[c] = (idx, cKDTree(xyz[idx, :2]))
        for i, ca in enumerate(comps):
            ia, ta = trees[ca]
            for cb in comps[i + 1:]:
                ib, tb = trees[cb]
                d, k = tb.query(xyz[ia, :2], distance_upper_bound=BRIDGE_GAP)
                ok = np.isfinite(d)
                if not ok.any():
                    continue
                order = np.argsort(d[ok])[:max_pairs]
                for a, b in zip(ia[ok][order], ib[k[ok][order]]):
                    dz = abs(xyz[a, 2] - xyz[b, 2])
                    if dz > MAX_STEP:
                        continue
                    seg = np.linspace(xyz[a], xyz[b], 8) + np.array([0.0, 0.0, 100.0])
                    c_min = float(np.min(self.clearance_points(seg)))
                    if c_min < WALK_CLEAR:
                        continue
                    length = float(np.linalg.norm(xyz[a] - xyz[b]))
                    for u, v in ((a, b), (b, a)):
                        rows.append(int(u))
                        cols.append(int(v))
                        w.append(length * (1.0 + (40.0 / c_min) ** 2))
                        plain.append(length)

    def propose_endpoints(self):
        """出発点・目的地の候補: 床の上を歩いて一番遠い2点(低い方を出発点)。

        一度ある点から一番遠い点 A を求め、A から一番遠い点 B を求める
        (グラフの直径の定番の近似)。玄関と一番奥の部屋がこれに当たることが
        多い。部屋が複数ある家では、ユーザーが候補から選ぶ前提の「既定値」。
        塗りつぶしの起点とつながっている床だけを対象にする。
        戻り値: (出発点の座標, 目的地の座標, 2点間の歩く距離cm)
        """
        xyz, _, lengths = self._walk_graph()
        if len(xyz) < 2:
            raise ValueError("歩ける床が見つからない")
        # 一番広くつながった床のかたまりから始める
        from scipy.sparse.csgraph import connected_components
        _, lab = connected_components(lengths, directed=False)
        big = np.bincount(lab).argmax()
        first = int(np.flatnonzero(lab == big)[0])
        d0 = dijkstra(lengths, indices=first)
        a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
        da = dijkstra(lengths, indices=a)
        b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
        # 一番遠い点は通路の隅(角)になりやすく、中心線の端が壁に寄る。
        # 近く(ENDPOINT_PULL 以内)で壁から一番離れた所(通路の中央)に寄せる
        clr = self._graph_clr

        def centered(i):
            near = np.flatnonzero(np.linalg.norm(xyz - xyz[i], axis=1) <= ENDPOINT_PULL)
            near = near[lab[near] == big]
            return xyz[near[np.argmax(clr[near])]]

        pa, pb = centered(a), centered(b)
        if pb[2] < pa[2]:
            pa, pb = pb, pa
        return pa, pb, float(da[b])

    def build_centerline(self, start_xyz, goal_xyz, ds=5.0, smooth_cm=60.0):
        """床の上を start から goal まで歩く経路を求め、中心線にする。

        壁から離れるほど安いコストで最短経路を探す(コーナーの内側に
        張り付かず、通路の中央寄りを通るように)。経路は移動平均で
        なめらかにし、弧長 ds ごとに resample する。
        """
        xyz, graph, _ = self._walk_graph()

        def nearest(p):
            p = np.asarray(p, dtype=float)
            return int(np.argmin(np.linalg.norm(xyz - p, axis=1)))

        s_node, g_node = nearest(start_xyz), nearest(goal_xyz)
        dist, pred = dijkstra(graph, indices=s_node, return_predecessors=True)
        if not np.isfinite(dist[g_node]):
            raise ValueError("出発点から目的地まで歩ける床がつながっていない")
        path = [g_node]
        while path[-1] != s_node:
            path.append(int(pred[path[-1]]))
        pts = xyz[path[::-1]]
        self.centerline = Centerline(pts, ds=ds, smooth_cm=smooth_cm)
        return self.centerline


class Centerline:
    """弧長 s で引ける中心線: 位置(x, y)・床の高さ・進行方向・勾配。"""

    def __init__(self, pts, ds=5.0, smooth_cm=60.0, corner_deg=20.0, corner_pad=40.0):
        pts = np.asarray(pts, dtype=float)
        # 弧長(水平)で等間隔に resample してから移動平均でなめらかにする
        seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        S = float(s[-1])
        si = np.arange(0.0, S + 1e-9, ds)
        res = np.column_stack([np.interp(si, s, pts[:, k]) for k in range(3)])
        win = max(1, int(round(smooth_cm / ds)))
        xy = np.column_stack([nd.uniform_filter1d(res[:, k], win, mode="nearest")
                              for k in range(2)])
        # なめらかにした後の弧長で張り直す
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s2 = np.concatenate([[0.0], np.cumsum(seg)])
        self.s_total = float(s2[-1])
        self.s = np.arange(0.0, self.s_total + 1e-9, ds)
        self.xy = np.column_stack([np.interp(self.s, s2, xy[:, k]) for k in range(2)])
        # 床の高さは段差をそのまま残す(なめらかにすると段の上に浮く)
        self.z = np.interp(self.s, s2, res[:, 2])
        d = np.gradient(self.xy, ds, axis=0)
        self.heading = np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))
        z_smooth = nd.uniform_filter1d(self.z, win, mode="nearest")
        self.slope_deg = np.degrees(np.arctan(np.gradient(z_smooth, ds)))
        # コーナー区間: 前後 smooth_cm/2 で進行方向が corner_deg 以上変わる所
        half = max(1, win // 2)
        turn = np.abs(np.roll(self.heading, -half) - np.roll(self.heading, half))
        turn[:half] = 0.0
        turn[-half:] = 0.0
        self.corners = []
        idx = np.nonzero(turn >= corner_deg)[0]
        if len(idx):
            groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
            for g in groups:
                s0 = max(0.0, self.s[g[0]] - corner_pad)
                s1 = min(self.s_total, self.s[g[-1]] + corner_pad)
                if self.corners and s0 <= self.corners[-1][1]:
                    self.corners[-1] = (self.corners[-1][0], s1)
                else:
                    self.corners.append((s0, s1))

    def _at(self, arr, s):
        return float(np.interp(s, self.s, arr))

    def xy_at(self, s):
        return np.array([self._at(self.xy[:, 0], s), self._at(self.xy[:, 1], s)])

    def z_at(self, s):
        return self._at(self.z, s)

    def heading_at(self, s):
        return self._at(self.heading, s)

    def slope_at(self, s):
        return self._at(self.slope_deg, s)

    def corner_at(self, s):
        """s を含むコーナー区間 (s0, s1)。コーナーでなければ None。"""
        for s0, s1 in self.corners:
            if s0 <= s <= s1:
                return s0, s1
        return None

    def project(self, x, y):
        """(x, y) に一番近い中心線上の弧長 s。"""
        return float(self.s[np.argmin(np.linalg.norm(self.xy - np.array([x, y]), axis=1))])
