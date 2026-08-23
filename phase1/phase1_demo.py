"""
Phase 1 最小デモ: L字廊下の曲がり角、箱単体 vs 箱+運搬者。

Phase 0 (2026-08-15) での実測:
  - 段ボール箱の設置面: 60cm x 42cm
  - 拡幅した曲がり角(箱単体はここを通過): 腕の幅 61cm / 52cm
  - 箱単体: 通過。同じ箱を1人で運ぶと: 失敗。

このスクリプトは単純な2D (x, y, theta) グリッドプランナー(離散化した
状態空間上のBFS)を組み、箱単体ならL字廊下を通り抜けられるが、
運搬者の体を取り付けると通れなくなることを示す。

このファイルの読み方(上から順に):
  1. パラメータ(cm) -- シナリオを決める数値。
  2. 幾何ヘルパー -- 「この点/形状は廊下の中か、何cmの余裕があるか」
     (クリアランス／符号付き距離の仕組み)。
  3. 可搬性オラクル -- place_humans / carriable_clearance: ある箱の
     姿勢に対して、壁にぶつからずに運搬者が立てる位置があるか、
     あるとしたら一番厳しい余裕は何cmか。
  4. プランナー -- 力任せのグリッド探索(RRTではなくグリッド上のBFSを
     使う理由は、下のDXの上のコメントを参照)。
  5. 描画 + __main__ -- 両方のケースを並べて描画し、数値結果
     (PASS/BLOCKED + 余裕cm)を表示する。

直接実行するとデモが見られる:
    python phase1_demo.py --carriers 2   # (既定) 前後1人ずつ
    python phase1_demo.py --carriers 1   # 1人で後方を持つ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---- パラメータ(cm)、Phase 0 の実測値から ----
# Phase 0 では曲がり角を61cm/52cmまで拡幅し、そこでは箱単体が通過した。
# その拡幅した隙間そのもの(実測値ちょうど、余裕なし)を厳密な2矩形の
# L字廊下としてモデル化すると、人が全くいなくても箱単体がぎりぎり
# 通らない結果になる(phase1の開発メモ参照) -- 実際の部屋には、
# 一番狭い箇所の先にこのモデルが捉えきれていない床の余白があった。
# このモデルで箱単体が通る、きりの良い数字まで4〜6cmほど緩めてある。
# 定性的な結果(箱単体は通り、運搬者を付けると通らない)は変わらない。
W1 = 65.0        # 廊下の腕の幅(横方向の腕)
W2 = 58.0        # 廊下の腕の幅(縦方向の腕)
ARM_LEN = 150.0  # 各廊下の腕の長さ

BOX_L = 60.0     # 箱の設置面、長辺
BOX_W = 42.0     # 箱の設置面、短辺

HUMAN_R = 20.0     # 運搬者の体を表すカプセルの半径 (CLAUDE.md: 0.2m)
# 把持点から運搬者の体の中心までの腕の長さ。
# Phase 0ではまだ未計測 -- 実測値待ちの仮置きで、箱や廊下と
# 同じオーダーの値を選んでいる。
CARRY_ARM = 15.0
# 運搬者が取り得る、箱の中心線に対する横方向の立ち位置。
SIDE_OFFSETS = (-15.0, -7.5, 0.0, 7.5, 15.0)

# 箱を運ぶ人数: 2人(両端に1人ずつ -- 家具では通常のケース)、
# または1人(1人で後方の把持点を持つ -- 例えばこの段ボール箱を
# 1人で運ぶ場合)。下のオラクル関連の関数はすべてこのグローバル値を
# 既定にしつつ、明示的なnum_carriers=での上書きも受け付ける
# (w1/w2/arm_lenと同じパターン)。
NUM_CARRIERS = 2

# START/GOALは各腕のモデル化された端からこの距離だけ内側に置く。
# BOX_L/2 + CARRY_ARM + HUMAN_R (ここでは30+15+20=65cm) 以上の
# 余裕が必要 -- start/goalの姿勢自体が運搬可能でなければならない。
# モデル化された腕の端は「廊下がまだ続いている」ことの代用であって
# 実際の壁ではないため、余裕が小さすぎると、実際には何もないのに
# 後方のカプセルがそこをはみ出してしまう。
END_MARGIN = 75.0
START = (ARM_LEN - END_MARGIN, W1 / 2, 0.0)
GOAL = (W2 / 2, ARM_LEN - END_MARGIN, np.pi / 2)


def in_free_space(x, y, w1=None, w2=None, arm_len=None):
    """(x, y) がL字廊下の内側にあればTrue。

    廊下は重なり合う2つの矩形の和集合でしかない: 横方向の腕
    (xは全長、yの幅はw1)と縦方向の腕(xの幅はw2、yは全長)。
    このファイル中のw1/w2/arm_lenパラメータは、Noneのままにすると
    すべてモジュールレベルのW1/W2/ARM_LENを既定値にする -- これに
    よって下のfind_critical_width()が、呼び出し箇所ひとつひとつに
    幅の引数を通さずに、別の廊下幅を試せるようになっている。
    """
    w1 = W1 if w1 is None else w1
    w2 = W2 if w2 is None else w2
    arm_len = ARM_LEN if arm_len is None else arm_len
    return (0 <= x <= arm_len and 0 <= y <= w1) or (0 <= x <= w2 and 0 <= y <= arm_len)


def _point_seg_dist(px, py, ax, ay, bx, by):
    """点(px, py)から線分(a, b)までの最短距離。

    「直線に投影してから線分の範囲にクランプする」という定番の方法:
    tはpの投影がaからbへの直線上のどこに落ちるかを、a→bの割合で
    表したもの。tを[0, 1]にクランプすることで、直線を端の先まで
    延長した場合ではなく、線分自体の上での最近点を保つ。
    """
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return float(np.hypot(px - ax, py - ay))  # a == b: 線分が点になる場合
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return float(np.hypot(px - cx, py - cy))


def dist_to_boundary(x, y, w1=None, w2=None, arm_len=None):
    """(x, y) からL字自由空間の一番近い壁までの距離。

    自由空間は2つの矩形の和集合で、その境界は
    (0,0) -> (arm_len,0) -> (arm_len,w1) -> (w2,w1) -> (w2,arm_len)
    -> (0,arm_len) という6角形になる。この6角形の辺までの距離は、
    内側の点にとってそのまま一番近い壁までのクリアランスになる。
    """
    w1 = W1 if w1 is None else w1
    w2 = W2 if w2 is None else w2
    arm_len = ARM_LEN if arm_len is None else arm_len
    verts = [(0, 0), (arm_len, 0), (arm_len, w1), (w2, w1), (w2, arm_len), (0, arm_len)]
    n = len(verts)
    return min(_point_seg_dist(x, y, *verts[i], *verts[(i + 1) % n]) for i in range(n))


def signed_clearance(x, y, w1=None, w2=None, arm_len=None):
    """内側なら一番近い壁までの余裕(+cm)、外側ならはみ出し量(-cm)。"""
    d = dist_to_boundary(x, y, w1, w2, arm_len)
    return d if in_free_space(x, y, w1, w2, arm_len) else -d


def shape_clearance(points, w1=None, w2=None, arm_len=None):
    """境界サンプル点の集合における最悪(最小)のクリアランス。

    「壁から形状全体までの距離」を、形状の境界上の有限個の点を
    チェックして一番悪いものを取ることで近似している(正確な
    点対多角形の距離を計算するのではなく)。安く済み、サンプル点を
    十分増やせば同じ答えに収束する。何点使うかはbox_sample_points/
    circle_boundary_pointsが決める。
    """
    return min(signed_clearance(px, py, w1, w2, arm_len) for px, py in points)


def rot(theta):
    """標準的な2D回転行列。箱ローカル座標(長辺方向がローカルx、
    短辺方向がローカルy)をワールド座標に変換するのに使う。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def box_corners(x, y, theta):
    """箱の輪郭を描くための、ワールド座標での4つの角。"""
    hl, hw = BOX_L / 2, BOX_W / 2
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return local @ rot(theta).T + np.array([x, y])


def box_sample_points(x, y, theta, nu=7, nv=4):
    """箱の設置面(辺と内部)を覆う、nu*nv個の点をワールド座標で。
    4つの角だけでなくこれを使うのは、2つの角の間で対角線が壁を
    突き抜けるケースも拾うため -- サンプル点を使う理由(正確な
    多角形距離ではなく)はshape_clearanceのdocstringを参照。"""
    hl, hw = BOX_L / 2, BOX_W / 2
    us = np.linspace(-hl, hl, nu)
    vs = np.linspace(-hw, hw, nv)
    local = np.array([[u, v] for u in us for v in vs])
    return local @ rot(theta).T + np.array([x, y])


def box_clearance(x, y, theta, w1=None, w2=None, arm_len=None):
    """この姿勢での箱単体(運搬者なし)のクリアランス(cm)。"""
    return shape_clearance(box_sample_points(x, y, theta), w1, w2, arm_len)


def circle_boundary_points(center, radius=HUMAN_R, n=16):
    """円周上に均等に並ぶn個の点 -- box_sample_pointsと同じ「形状の
    境界をサンプルする」トリックを、箱ではなく運搬者のカプセルに
    適用したもの。"""
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]


def place_humans(x, y, theta, side_offset, num_carriers=None):
    """与えられた横方向の立ち位置に対する、運搬者カプセルの中心。

    把持点は箱の端に置かれ、それぞれ箱の長軸方向にCARRY_ARMだけ
    外側に伸ばし、それに垂直な方向へ共通の横方向のずれ(side_offset)
    を加える -- これがCLAUDE.mdのcarriable(q, h)オラクルで、hが
    SIDE_OFFSETSの範囲を動く。

    num_carriers=2 (既定): 両端に1人ずつ -- (front, back) を返す。
    num_carriers=1: 箱の後方を1人だけで持つ -- (back,) という
    1要素タプルを返す。呼び出す側がどちらのケースかを気にせず
    ループできるよう、常に(x, y)点の1要素または2要素タプルを返す。
    """
    num_carriers = NUM_CARRIERS if num_carriers is None else num_carriers
    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-d[1], d[0]])
    c = np.array([x, y])
    hl = BOX_L / 2

    grip_back = c - d * hl
    h_back = tuple(grip_back - d * CARRY_ARM + n * side_offset)
    if num_carriers == 1:
        return (h_back,)

    grip_front = c + d * hl
    h_front = tuple(grip_front + d * CARRY_ARM + n * side_offset)
    return (h_front, h_back)


def carriable_clearance(x, y, theta, w1=None, w2=None, arm_len=None, num_carriers=None):
    """許容される運搬者の立ち位置全体の中で達成できる最良のクリアランス(cm)。

    SIDE_OFFSETSの立ち位置をすべて試し、一番有利なものを残す。
    つまりこれが答えるのは「箱の隣に立つ何らかの方法が機能するか、
    機能するならどれだけの余裕があるか」であって、「この1つの
    立ち位置が機能するか」ではない。(clearance_cm, best_side_offset)
    を返す。best_side_offsetは、箱自体が収まらない場合(運搬者の
    配置自体が意味をなさない場合)はNoneになる。
    """
    bc = box_clearance(x, y, theta, w1, w2, arm_len)
    if bc < 0:
        return bc, None
    best = -np.inf
    best_offset = None
    for off in SIDE_OFFSETS:
        capsule_centers = place_humans(x, y, theta, off, num_carriers)
        capsule_clearances = [
            shape_clearance(circle_boundary_points(p), w1, w2, arm_len)
            for p in capsule_centers
        ]
        # 全体の配置(箱+すべての運搬者)は、一番厳しい箇所の分しか
        # 良くならない。
        cand = min([bc] + capsule_clearances)
        if cand > best:
            best = cand
            best_offset = off
    return best, best_offset


def best_human_positions(x, y, theta, num_carriers=None):
    """描画専用: 最良の立ち位置での運搬者カプセルの中心。"""
    _, offset = carriable_clearance(x, y, theta, num_carriers=num_carriers)
    if offset is None:
        offset = 0.0  # 箱自体がすでに失敗しているので立ち位置の選択自体は無意味、何か描画するためだけの値
    return place_humans(x, y, theta, offset, num_carriers)


def collision_free(x, y, theta, with_human, w1=None, w2=None, arm_len=None, num_carriers=None):
    """プランナーが各グリッドセルに対して使う、唯一のYes/No判定。

    with_human=False: 箱単体だけでこの姿勢を占有できるか?
    with_human=True: 箱と運搬者を合わせてこの姿勢を占有できるか
    -- つまりcarriable_clearanceが達成できる最良のクリアランスが
    0以上か? CLAUDE.mdの「衝突判定器を可搬性オラクルに差し替える」
    という発想が実際に起きているのはここだけで、この関数より下
    (グリッド/BFSプランナー)はどちらのケースかを知らないし
    気にもしない。ただ設定されたcollision_free()を呼ぶだけ。
    """
    if with_human:
        clearance, _ = carriable_clearance(x, y, theta, w1, w2, arm_len, num_carriers)
        return clearance >= 0
    return box_clearance(x, y, theta, w1, w2, arm_len) >= 0


def angle_wrap(a):
    """角度を(-pi, pi]に正規化する。例えば359度と1度を比べたときに
    358度ではなく2度の差になるようにする。"""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- グリッドベースのプランナー ----
# 廊下は箱に対して狭く(ランダムな(x,y,theta)状態のうち衝突フリーな
# ものは1〜3%程度しかない)、単純なランダムサンプリングのRRTでは
# 木の成長が遅すぎて一晩で確実なデモにならない。代わりに(x, y,
# theta)を細かいグリッドに離散化し、それを網羅的に探索する(BFS)。
# これはresolution-complete: グリッドが「経路なし」と言うなら、
# それはこの解像度では本当に経路が存在しないからであり、サンプリング
# の運の悪さではない。時間ができたら本物のRRT-Connectに戻す
# (CLAUDE.mdのPhase 1参照)。
DX = 4.0
DTHETA = np.radians(15)
NTH = int(round(2 * np.pi / DTHETA))


def build_grid(with_human, w1=None, w2=None, arm_len=None, num_carriers=None):
    """3次元の真偽値配列free[i, j, k]を事前計算する: グリッドセル
    (x=xs[i], y=ys[j], theta=thetas[k])は衝突フリーか? これを
    探索中にcollision_freeを呼ぶのではなく最初に1回まとめてやる
    ことで、下のgrid_bfsの近傍探索が単なる配列インデックス参照
    で済むようになる。
    """
    arm_len = ARM_LEN if arm_len is None else arm_len
    nx = int(round(arm_len / DX)) + 1
    ny = nx
    xs = np.linspace(0, arm_len, nx)
    ys = np.linspace(0, arm_len, ny)
    thetas = -np.pi + DTHETA * np.arange(NTH)
    free = np.zeros((nx, ny, NTH), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if not in_free_space(x, y, w1, w2, arm_len):
                continue  # そもそも廊下の外; ここではどのthetaもブロックされる
            for k, th in enumerate(thetas):
                free[i, j, k] = collision_free(x, y, th, with_human, w1, w2, arm_len, num_carriers)
    return xs, ys, thetas, free


def nearest_index(xs, ys, thetas, state):
    """連続値の(x, y, theta)状態を、一番近いグリッドセルのインデックス
    (i, j, k)に丸める。START/GOALがグリッド上のどこに来るかを
    求めるのに使う。angle_wrapによって、theta方向の探索が
    -pi/+piの境目で混乱しないようにしている。
    """
    x, y, th = state
    i = int(np.argmin(np.abs(xs - x)))
    j = int(np.argmin(np.abs(ys - y)))
    k = int(np.argmin(np.abs(angle_wrap(thetas - th))))
    return i, j, k


def _reconstruct(xs, ys, thetas, prev, start_idx, end_idx):
    """BFSの親ポインタ(prev)をend_idxからstart_idxまで逆向きに辿り、
    reverseしてstart→endの順にする。途中でグリッドのインデックスを
    (x, y, theta)の状態に戻す。"""
    path_idx = [end_idx]
    while path_idx[-1] != start_idx:
        path_idx.append(prev[path_idx[-1]])
    path_idx.reverse()
    return [(xs[i], ys[j], thetas[k]) for i, j, k in path_idx]


def grid_bfs(start, goal, with_human, track_best_effort=False, w1=None, w2=None, arm_len=None, num_carriers=None):
    """build_gridで作った(x, y, theta)グリッド上の幅優先探索。

    通常は(path, free, xs, ys, thetas)を返す: pathはstartからgoal
    までの(x, y, theta)状態のリスト、経路が存在しなければNone。

    track_best_effort=Trueの場合は代わりに(path, best_effort_path,
    free, xs, ys, thetas)を返す: best_effort_pathは常に埋まっている
    (pathがNoneのときでもBFSがたどり着けた一番ゴールに近い状態)。
    これによって、単なる失敗を報告するのではなく、運搬者が
    「どこで」詰まるかをデモで示せるようになる。
    """
    xs, ys, thetas, free = build_grid(with_human, w1, w2, arm_len, num_carriers)
    nx, ny = len(xs), len(ys)
    start_idx = nearest_index(xs, ys, thetas, start)
    goal_idx = nearest_index(xs, ys, thetas, goal)
    if not free[start_idx]:
        return (None, free, xs, ys, thetas) if not track_best_effort else (None, None, free, xs, ys, thetas)
    if not free[goal_idx] and not track_best_effort:
        # ゴールセルがブロックされている以上そこで終わる経路は作れない;
        # 呼び出し側がbest-effortを求めている場合だけ、下の全探索を
        # する価値がある。
        return None, free, xs, ys, thetas

    from collections import deque
    visited = np.zeros_like(free, dtype=bool)
    prev = {}  # セル -> BFSがそのセルに到達する元になったセル(経路の逆算用)
    q = deque([start_idx])
    visited[start_idx] = True
    # 3D (x, y, theta) グリッドにおける26個の近傍セルすべて
    # (3*3*3 - 1、「まったく動かない」を除く)。BFSはこれらの移動を
    # すべて同じコスト1として扱うので、あるセルに最初に到達した
    # ときの経路が、(グリッドのステップ数として)最短の経路になる。
    neighbors = [(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 for dk in (-1, 0, 1) if not (di == 0 and dj == 0 and dk == 0)]

    gx, gy = goal[0], goal[1]
    best_idx = start_idx
    best_dist = np.hypot(xs[start_idx[0]] - gx, ys[start_idx[1]] - gy)

    while q:
        cur = q.popleft()
        if cur == goal_idx:
            path = _reconstruct(xs, ys, thetas, prev, start_idx, cur)
            if track_best_effort:
                return path, path, free, xs, ys, thetas
            return path, free, xs, ys, thetas
        ci, cj, ck = cur
        for di, dj, dk in neighbors:
            # thetaは(NTHを法として)ぐるっと繋がっているが、xとyは
            # 繋がっていない。そのため下で明示的に
            # 0 <= ni < nx / 0 <= nj < ny の範囲チェックをしている。
            ni, nj, nk = ci + di, cj + dj, (ck + dk) % NTH
            if 0 <= ni < nx and 0 <= nj < ny and free[ni, nj, nk] and not visited[ni, nj, nk]:
                visited[ni, nj, nk] = True
                prev[(ni, nj, nk)] = cur
                q.append((ni, nj, nk))
                if track_best_effort:
                    # (単純なxy距離で)これまでにゴールへ一番近づけた
                    # 訪問済みセルを記録しておく。探索がゴールに
                    # 到達しないまま尽きた場合に備えて。
                    d = np.hypot(xs[ni] - gx, ys[nj] - gy)
                    if d < best_dist:
                        best_dist = d
                        best_idx = (ni, nj, nk)

    # キューが空になってもgoal_idxが見つからなかった: この解像度では
    # 経路が存在しない。要求されていれば、それでもbest-effortの経路を返す。
    if track_best_effort:
        best_path = _reconstruct(xs, ys, thetas, prev, start_idx, best_idx)
        return None, best_path, free, xs, ys, thetas
    return None, free, xs, ys, thetas


def path_min_clearance(path, with_human, num_carriers=None):
    """見つかった経路上で最も厳しい(最小の)クリアランス(cm)。"""
    if with_human:
        return min(carriable_clearance(*s, num_carriers=num_carriers)[0] for s in path)
    return min(box_clearance(*s) for s in path)


def find_critical_width(lo=40.0, hi=70.0, iters=10, num_carriers=None):
    """箱単体のプランナーはまだ経路を見つけられるが、可搬性オラクルは
    見つけられなくなる、単一の(W1=W2=w)廊下幅を二分探索する。

    各試行ごとにモジュールレベルのW1/W2グローバル変数を一時的に
    上書きし(build_grid/in_free_spaceはw1/w2が明示的に渡されない
    場合これらを既定値として使う)、終わったら元に戻す。
    """
    global W1, W2
    orig_w1, orig_w2 = W1, W2
    results = []
    found_w = None
    try:
        for _ in range(iters):
            w = (lo + hi) / 2.0
            start = (ARM_LEN - END_MARGIN, w / 2.0, 0.0)
            goal = (w / 2.0, ARM_LEN - END_MARGIN, np.pi / 2)
            W1, W2 = w, w
            path_free, *_ = grid_bfs(start, goal, with_human=False)
            path_human, *_ = grid_bfs(start, goal, with_human=True, num_carriers=num_carriers)
            ok_free = path_free is not None
            ok_human = path_human is not None
            results.append((w, ok_free, ok_human))
            if ok_free and not ok_human:
                found_w = w
                break
            elif not ok_free:
                lo = w
            else:
                hi = w
    finally:
        W1, W2 = orig_w1, orig_w2
    return found_w, results


def draw_env(ax, title):
    """L字廊下を、重なり合う2つの灰色矩形として描く -- in_free_space()
    が判定に使っているのと同じ2つの矩形。"""
    ax.add_patch(patches.Rectangle((0, 0), ARM_LEN, W1, facecolor='#e8e8e8', edgecolor='none', zorder=0))
    ax.add_patch(patches.Rectangle((0, 0), W2, ARM_LEN, facecolor='#e8e8e8', edgecolor='none', zorder=0))
    ax.set_xlim(-10, ARM_LEN + 10)
    ax.set_ylim(-10, ARM_LEN + 10)
    ax.set_aspect('equal')
    ax.set_title(title)


def draw_box(ax, x, y, theta, color='tab:blue', alpha=0.5, lw=1.0):
    corners = box_corners(x, y, theta)
    ax.add_patch(patches.Polygon(corners, closed=True, facecolor=color, edgecolor='k', alpha=alpha, linewidth=lw, zorder=2))


def draw_carriers(ax, x, y, theta, color='tab:red', alpha=0.4, num_carriers=None):
    """最良の横位置における、すべての運搬者カプセル(num_carriersに応じて1個か2個)。"""
    for c in best_human_positions(x, y, theta, num_carriers):
        ax.add_patch(patches.Circle(c, HUMAN_R, facecolor=color, edgecolor='k', alpha=alpha, zorder=1))


def run_case(ax, with_human, label, num_carriers=None):
    """START→GOALを1ケース分(運搬者ありまたはなし)探索し、axに
    描画する: 経路が見つかればその上に薄い「残像」の箱(と運搬者)を
    並べ、見つからなければstart/goalの姿勢だけを描く。"""
    path, free, xs, ys, thetas = grid_bfs(START, GOAL, with_human=with_human, num_carriers=num_carriers)
    found = path is not None
    draw_env(ax, f"{label}: {'PASS (path found)' if found else 'BLOCKED (no path exists at this resolution)'}")

    if found:
        step = max(1, len(path) // 12)  # 残像は12枚程度だけ描く。全部描くと密集して読めない
        for s in path[::step]:
            draw_box(ax, *s, alpha=0.15)
            if with_human:
                draw_carriers(ax, *s, alpha=0.10, num_carriers=num_carriers)
        draw_box(ax, *path[0], color='tab:green', alpha=0.9)
        draw_box(ax, *path[-1], color='tab:blue', alpha=0.9)
        if with_human:
            draw_carriers(ax, *path[0], color='tab:green', alpha=0.5, num_carriers=num_carriers)
            draw_carriers(ax, *path[-1], color='tab:blue', alpha=0.5, num_carriers=num_carriers)
    else:
        draw_box(ax, *START, color='tab:green', alpha=0.9)
        draw_box(ax, *GOAL, color='tab:blue', alpha=0.4)
        if with_human:
            draw_carriers(ax, *START, color='tab:green', alpha=0.5, num_carriers=num_carriers)
            draw_carriers(ax, *GOAL, color='tab:blue', alpha=0.3, num_carriers=num_carriers)
    return found, path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carriers", type=int, choices=(1, 2), default=NUM_CARRIERS,
        help="How many people carry the box: 1 (solo, trailing behind) "
             "or 2 (one at each end). Default: %(default)s.")
    args = parser.parse_args()
    num_carriers = args.carriers
    carrier_label = "1 carrier" if num_carriers == 1 else "2 carriers"

    # --- デモ1: Phase 0で実測した固定シナリオ(廊下65x58cm) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    found1, path1 = run_case(axes[0], with_human=False, label="Box only")
    found2, path2 = run_case(axes[1], with_human=True, label=f"Box + {carrier_label}", num_carriers=num_carriers)

    fig.suptitle("Odoriba Phase 1: L-corridor, box vs box+human "
                  f"(corridor {W1:.0f}x{W2:.0f}cm, box {BOX_L:.0f}x{BOX_W:.0f}cm, "
                  f"carrier r={HUMAN_R:.0f}cm, {carrier_label})")
    fig.tight_layout()
    out_path = "phase1_demo.png"
    fig.savefig(out_path, dpi=150)

    print(f"carriers: {num_carriers}")
    print(f"box only:    {'PASS' if found1 else 'BLOCKED'}"
          + (f" (min clearance {path_min_clearance(path1, False):.1f}cm)" if found1 else ""))
    print(f"box+carrier: {'PASS' if found2 else 'BLOCKED'}"
          + (f" (min clearance {path_min_clearance(path2, True, num_carriers):.1f}cm)" if found2 else ""))
    print(f"saved figure to {out_path}")

    # --- デモ2: 単一の対称な幅を振って、「箱は通るが運搬者は通らない」
    # が最初に現れるちょうどのcmを探す(探索そのものの説明は
    # find_critical_widthのdocstring参照)。---
    print("\nsearching for the critical corridor width (box passes, carrier blocked)...")
    w, results = find_critical_width(num_carriers=num_carriers)
    if w is not None:
        print(f"廊下幅 {w:.1f}cm / 箱 {BOX_L:.0f}x{BOX_W:.0f}cm / 人 r={HUMAN_R:.0f}cm, arm={CARRY_ARM:.0f}cm, {carrier_label}")
        print("  自由剛体  : 通る")
        print("  人あり    : 通らない")
    else:
        print("no width in the search range separated the two cases; see `results` for the raw sweep")
    for w_, ok_free, ok_human in results:
        print(f"  w={w_:5.1f}cm  box_only={'PASS' if ok_free else 'BLOCK'}  "
              f"with_carrier={'PASS' if ok_human else 'BLOCK'}")

    plt.show()
