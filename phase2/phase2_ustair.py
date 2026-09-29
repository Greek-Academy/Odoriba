"""
Phase 2 拡張: 折り返し(U字)階段(上り6段 -> 踊り場 -> 180度折り返して上り6段)。

日本の戸建てで最も多い階段の形。L字(90度)より踊り場での回転が
大きく(180度)、見せたい差がよりはっきり出る:

    家具単体(自由に浮遊する剛体)なら立てて回せるが、
    運搬者を付けると踊り場で折り返せない。

L字(phase2_lstair.py)との分担:
  - 家具形状・可搬性オラクル・RRT-Connect は phase2_lstair / phase2_demo を
    そのまま使う(オラクルには歩行面関数 floor_fn だけ差し替えて渡す)
  - このファイルが持つのは幾何だけ: 環境の直方体、歩行面の高さ、中心線、
    骨格誘導サンプラー、ボトルネック掃引

座標系: 下の廊下を+y方向に進み、踊り場で+x方向へ横切り、
-y方向に折り返して上る。上下のフライトの間は壁(厚さWALL)。

    y
    ^   +---------------------------+
    |   |          踊り場            |
    |   +--------+------+-----------+
    |   | フライト1 | 壁   | フライト2 |
    |   |  (上る↑) |      |  (上る↓)  |
    |   | 下廊下    |      | 上廊下    |
    +---+--------+------+-----------+--> x

簡略化: 上廊下の床スラブは下廊下の上に重ならない(平面的に横並び)。
実際の住宅では2階の床が下廊下の頭上に来ることがあるが、その頭上の
制約は天井高 CEIL_CLEAR で代表させる。

実行(リポジトリの phase2/ で):
    python phase2_ustair.py --max-iter 800 --skip-sweep   # 動作確認用
    python phase2_ustair.py                               # 本番(数分かかる)
"""

import numpy as np

import phase2_lstair as L

# ---- パラメータ(cm) ----
# L字と同じ「ごく普通の戸建て」の仮値。踊り場は幅W x 奥行W の正方形2枚
# と壁の分をつないだ長方形(2W+WALL) x W になる。
STAIR_WIDTH = 90.0    # 廊下・各フライトの幅(=踊り場の奥行き)
WALL = 10.0           # 上下フライトの間の壁の厚さ
RISE = 17.0           # 蹴上げ
TREAD = 26.0          # 踏み面
N_STEPS1 = 6          # 下側フライトの段数
N_STEPS2 = 6          # 上側フライトの段数
BASE_D = 340.0        # 下の廊下の奥行き(START用の余白)
TOP_D = 340.0         # 上の廊下の奥行き(GOAL用の余白)
CEIL_CLEAR = 220.0    # 各フライト最上段の踏み面から天井までの高さ


def _derive():
    """寸法パラメータから派生量(座標・中心線)を作り直す。"""
    global FL1_Y0, FL1_Y1, LAND_Y1, LAND_Z, X_R0, X_R1, FL2_Y1, TOP_Y0, TOP_Z
    global CX1, CX2, LAND_YC, S1, S2, S_TOTAL, CORNER_S0, CORNER_S1, STAIR_ANGLE_DEG
    FL1_Y0 = BASE_D                     # フライト1の開始y
    FL1_Y1 = BASE_D + N_STEPS1 * TREAD  # フライト1の終了y = 踊り場の開始y
    LAND_Y1 = FL1_Y1 + STAIR_WIDTH      # 踊り場の終了y(奥の壁)
    LAND_Z = N_STEPS1 * RISE            # 踊り場の床の高さ
    X_R0 = STAIR_WIDTH + WALL           # 右側(フライト2・上廊下)の左端x
    X_R1 = X_R0 + STAIR_WIDTH           # 右側の右端x = 踊り場の右端
    FL2_Y1 = FL1_Y1 - N_STEPS2 * TREAD  # フライト2の終了y(-y方向に上る)
    TOP_Y0 = FL2_Y1 - TOP_D             # 上廊下の終端y
    TOP_Z = LAND_Z + N_STEPS2 * RISE    # 上廊下の床の高さ

    # 中心線: (CX1, 0) -> (CX1, LAND_YC) -> (CX2, LAND_YC) -> (CX2, TOP_Y0)
    CX1 = STAIR_WIDTH / 2
    CX2 = X_R0 + STAIR_WIDTH / 2
    LAND_YC = FL1_Y1 + STAIR_WIDTH / 2
    S1 = LAND_YC                        # 1つ目の折れ点までの弧長
    S2 = S1 + (CX2 - CX1)               # 2つ目の折れ点までの弧長
    S_TOTAL = S2 + (LAND_YC - TOP_Y0)

    # コーナー(踊り場)ゾーン: 踊り場に入る少し手前から、出た少し先まで。
    # この範囲では回転ヒント(ヨー一様+立て上げピッチ)を撒く。
    CORNER_S0 = FL1_Y1 - 40.0
    CORNER_S1 = S2 + (LAND_YC - FL1_Y1) + 40.0
    STAIR_ANGLE_DEG = np.degrees(np.arctan2(RISE, TREAD))


_derive()


def configure(width=None, rise=None, tread=None, n_steps1=None, n_steps2=None,
              wall=None, base_d=None, top_d=None, ceil_clear=None):
    """折り返し階段の寸法を数値で設定し、派生量とSTART/GOALを作り直す。
    指定しない引数は現在値を維持する(L.configureと同じ使い方)。"""
    global STAIR_WIDTH, RISE, TREAD, N_STEPS1, N_STEPS2, WALL, BASE_D, TOP_D, CEIL_CLEAR
    if width is not None: STAIR_WIDTH = float(width)
    if rise is not None: RISE = float(rise)
    if tread is not None: TREAD = float(tread)
    if n_steps1 is not None: N_STEPS1 = int(n_steps1)
    if n_steps2 is not None: N_STEPS2 = int(n_steps2)
    if wall is not None: WALL = float(wall)
    if base_d is not None: BASE_D = float(base_d)
    if top_d is not None: TOP_D = float(top_d)
    if ceil_clear is not None: CEIL_CLEAR = float(ceil_clear)
    _derive()
    make_start_goal()


# ---- 環境 ----

def _aabb(x0, x1, y0, y1, z0, z1):
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    h = np.array([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2])
    return (c, np.eye(3), h)


def build_ustairs():
    """折り返し階段の環境を返す。

    戻り値: (outer, obstacles)  ※L.build_lstairsと同じ形式
      outer = 直方体3つのリスト(合併が自由空間の輪郭)
        A: 下廊下+フライト1(左の棟)
        B: 踊り場(横長。上は上廊下の天井まで吹き抜け)
        C: フライト2+上廊下(右の棟)
      上下フライトの間の壁は、どの外枠にも含まれないので自動的に壁になる。
      obstacles = ステップ・スラブの直方体リスト
    """
    top_ceil = TOP_Z + CEIL_CLEAR
    outer = [
        _aabb(0, STAIR_WIDTH, 0, LAND_Y1, 0, LAND_Z + CEIL_CLEAR),
        _aabb(0, X_R1, FL1_Y1, LAND_Y1, 0, top_ceil),
        _aabb(X_R0, X_R1, TOP_Y0, LAND_Y1, 0, top_ceil),
    ]

    obstacles = []
    # フライト1(+y方向へ上る): i段目のy範囲の下を実体で塞ぐ
    for i in range(N_STEPS1):
        y0 = FL1_Y0 + TREAD * i
        obstacles.append(_aabb(0, STAIR_WIDTH, y0, y0 + TREAD, 0, RISE * (i + 1)))
    # 踊り場の床下スラブ(左右の棟をまたぐ)
    obstacles.append(_aabb(0, X_R1, FL1_Y1, LAND_Y1, 0, LAND_Z))
    # フライト2(-y方向へ上る)
    for i in range(N_STEPS2):
        y1 = FL1_Y1 - TREAD * i
        obstacles.append(_aabb(X_R0, X_R1, y1 - TREAD, y1, 0, LAND_Z + RISE * (i + 1)))
    # 上廊下の床下スラブ
    obstacles.append(_aabb(X_R0, X_R1, TOP_Y0, FL2_Y1, 0, TOP_Z))
    return outer, obstacles


def floor_z(x, y):
    """(x, y) の真下の歩行面の高さ。歩ける場所でなければNone。

    可搬性オラクル(L.state_valid等)には floor_fn=floor_z として渡す。
    """
    if FL1_Y1 <= y <= LAND_Y1 and 0.0 <= x <= X_R1:
        return LAND_Z
    if 0.0 <= x <= STAIR_WIDTH and 0.0 <= y < FL1_Y1:
        if y < FL1_Y0:
            return 0.0
        i = min(int((y - FL1_Y0) // TREAD), N_STEPS1 - 1)
        return RISE * (i + 1)
    if X_R0 <= x <= X_R1 and TOP_Y0 <= y < FL1_Y1:
        if y < FL2_Y1:
            return TOP_Z
        i = min(int((FL1_Y1 - y) // TREAD), N_STEPS2 - 1)
        return LAND_Z + RISE * (i + 1)
    return None


# ---- 骨格(中心線) ----

def skeleton_xy(s):
    """弧長s(0..S_TOTAL)における中心線上の点(x, y)。"""
    if s <= S1:
        return CX1, s
    if s <= S2:
        return CX1 + (s - S1), LAND_YC
    return CX2, LAND_YC - (s - S2)


def skeleton_heading(s):
    """弧長sでの進行方向のヨー(度)。フライト1=90, 踊り場=0, フライト2=-90。"""
    if s <= S1:
        return 90.0
    if s <= S2:
        return 0.0
    return -90.0


def skeleton_s(x, y):
    """(x, y) を中心線に射影したときの弧長s(進捗の測定用)。"""
    t1 = np.clip(y, 0.0, S1)
    d1 = np.hypot(x - CX1, y - t1)
    t2 = np.clip(x, CX1, CX2)
    d2 = np.hypot(x - t2, y - LAND_YC)
    t3 = np.clip(y, TOP_Y0, LAND_YC)
    d3 = np.hypot(x - CX2, y - t3)
    cands = ((d1, t1), (d2, S1 + (t2 - CX1)), (d3, S2 + (LAND_YC - t3)))
    return float(min(cands, key=lambda c: c[0])[1])


def on_flight(s):
    """弧長sがフライト(段の上)にあるか。勾配ピッチのヒントに使う。"""
    return (FL1_Y0 <= s <= FL1_Y1) or \
           (S2 + (LAND_YC - FL1_Y1) <= s <= S2 + (LAND_YC - FL2_Y1))


def walk_z(s):
    """中心線に沿った弧長sでの歩行面の高さ(サンプリング誘導用)。"""
    z = floor_z(*skeleton_xy(s))
    return 0.0 if z is None else z


def min_center_z(s, pitch_deg):
    """弧長sで長軸をpitchだけ傾けたとき、家具の最下点が歩行面に触れる
    ときの中心高さ(L.min_center_zの折り返し版)。"""
    pr = np.radians(abs(pitch_deg))
    return walk_z(s) + (L.FURN_L / 2) * np.sin(pr) + (L.FURN_H / 2) * np.cos(pr)


def where_label(s):
    """弧長sの場所の英語ラベル(図の注記用)。"""
    return "landing" if FL1_Y1 <= s <= S2 + (LAND_YC - FL1_Y1) else f"s={s:.0f}cm"


# ---- START / GOAL ----

def make_start_goal():
    """START(下廊下、長軸+y)とGOAL(上廊下、長軸-y)を作り直す。"""
    global START, GOAL
    START = (np.array([CX1, 170.0, L.FURN_H / 2]), L._pose(90.0, 0.0).as_quat())
    GOAL = (np.array([CX2, TOP_Y0 + 170.0, TOP_Z + L.FURN_H / 2]),
            L._pose(-90.0, 0.0).as_quat())


make_start_goal()


# ---- 骨格誘導サンプラー / validator ----

def make_sampler(with_human, p_guided=0.8, pos_noise=10.0, rot_noise_deg=8.0):
    """折り返し階段用の骨格誘導サンプラー(L.make_samplerと同じ構造)。

    - 位置: 中心線上の弧長sを一様に引き、横方向・高さにノイズ
    - 姿勢: フライト上では長軸を進行方向に向けて勾配ピッチ。
      踊り場ゾーンではヨーを-105..105度で一様(フライト1の90度から
      フライト2の-90度まで、0度を経由して180度回しきれるように)。
      家具単体は垂直近くまで「立てて回す」解を重点的に撒く
    - 確率(1 - p_guided)では完全ランダム(誘導が思いつかない解への保険)
    """
    max_pitch = L.MAX_TILT_DEG if with_human else 92.0

    def sampler(rng):
        if rng.random() >= p_guided:
            pos = rng.uniform([0, TOP_Y0, 0], [X_R1, LAND_Y1, TOP_Z + CEIL_CLEAR])
            return pos, L.g3.random_quaternion(rng)
        s = rng.uniform(0.0, S_TOTAL)
        if CORNER_S0 <= s <= CORNER_S1:
            yaw = rng.uniform(-105.0, 105.0)
            if with_human:
                pitch = rng.uniform(0.0, max_pitch)
            else:
                pitch = rng.uniform(55.0, 92.0) if rng.random() < 0.6 \
                    else rng.uniform(0.0, 55.0)
        else:
            base_pitch = STAIR_ANGLE_DEG if on_flight(s) else 0.0
            yaw = skeleton_heading(s) + rng.normal(0.0, rot_noise_deg)
            pitch = base_pitch + rng.normal(0.0, rot_noise_deg)
        roll = rng.normal(0.0, rot_noise_deg * 0.5)
        quat = L._pose(yaw, pitch, roll).as_quat()

        x, y = skeleton_xy(s)
        z = min_center_z(s, pitch) + abs(rng.normal(0.0, 12.0))
        pos = np.array([x, y, z]) + rng.normal(0.0, pos_noise, size=3) * np.array([1, 1, 0.3])
        return pos, quat

    return sampler


def state_valid(pos, quat, with_human, outer, obstacles, num_carriers=None):
    """rrt_connectのvalidator。L字のオラクルに折り返しの歩行面を渡すだけ。"""
    return L.state_valid(pos, quat, with_human, outer, obstacles, num_carriers,
                         floor_fn=floor_z)


def best_effort_path(tree):
    """RRTが失敗したとき、start側ツリーで中心線上を一番先まで進めたノード
    までの経路(L.best_effort_pathの折り返し版)。"""
    progress = [skeleton_s(n.pos[0], n.pos[1]) for n in tree]
    i = int(np.argmax(progress))
    path = []
    while i is not None:
        n = tree[i]
        path.append((n.pos, n.quat))
        i = n.parent
    path.reverse()
    return path


# ---- ボトルネック掃引 ----

def sweep_capacity(outer, obstacles, num_carriers=None, ds=15.0, with_human=True):
    """中心線上の各弧長sで、粗い姿勢グリッドの中で達成できる最良の
    クリアランス(cm)を返す: [(s, capacity_cm, best_pose), ...]。
    L.sweep_capacityの折り返し版(姿勢グリッドの考え方は同じ)。

    capacityが負の位置は「サンプルした姿勢のどれをとっても|capacity| cm
    足りない」ことを意味する(姿勢グリッドの解像度の範囲で)。
    """
    max_pitch = L.MAX_TILT_DEG if with_human else 90.0
    pitches = [t for t in (0.0, 15.0, 30.0, 45.0, 55.0, 70.0, 85.0, 90.0) if t <= max_pitch]
    laterals = (-10.0, 0.0, 10.0)
    z_extras = (0.0, 12.0, 30.0)
    corner_yaws = tuple(np.arange(-90.0, 90.0 + 1e-9, 22.5))

    # 端に近すぎるsはモデル化した廊下の端からはみ出すだけなので掃引しない
    s_margin = L.FURN_L / 2 + L.p.CARRY_ARM + L.p.HUMAN_R + 10.0
    results = []
    for s in np.arange(s_margin, S_TOTAL - s_margin + 1e-9, ds):
        x0, y0 = skeleton_xy(s)
        in_corner = CORNER_S0 <= s <= CORNER_S1
        yaws = corner_yaws if in_corner else (skeleton_heading(s),)
        # 横方向 = 中心線に直交する向き(踊り場の横断中はy、フライト上はx)
        along_x = S1 < s <= S2
        best = -np.inf
        best_pose = None
        for yaw in yaws:
            for pitch in pitches:
                quat = L._pose(yaw, pitch).as_quat()
                z_base = min_center_z(s, pitch)
                for lat in laterals:
                    x = x0 + (0.0 if along_x else lat)
                    y = y0 + (lat if along_x else 0.0)
                    for dz in z_extras:
                        pos = np.array([x, y, z_base + dz])
                        if with_human:
                            c, _ = L.carriable_clearance_lstair(
                                pos, quat, outer, obstacles, num_carriers, floor_fn=floor_z)
                        else:
                            c = L.furniture_clearance(pos, quat, outer, obstacles)
                        if c > best:
                            best = c
                            best_pose = (pos.tolist(), quat.tolist())
        results.append((float(s), float(best), best_pose))
    return results


def bottleneck_entry(sweep):
    """掃引結果から最悪の位置を結果JSONの1項目(dict)にまとめる。"""
    finite = [r for r in sweep if np.isfinite(r[1])]
    s_min, c_min, pose_min = min(finite, key=lambda r: r[1])
    # 余裕が負の連続区間(=詰まる場所)を数える。折り返しでは踊り場の
    # 入口と出口の2つの90度旋回がそれぞれ別の詰まり所になり得る。
    zones, run = [], None
    for s, c, _ in sweep:
        if c < 0:
            run = [s, s, c] if run is None else [run[0], s, min(run[2], c)]
        elif run is not None:
            zones.append(run)
            run = None
    if run is not None:
        zones.append(run)
    return {"s": float(s_min), "capacity_cm": float(c_min),
            "blocked_zones": [[float(a), float(b), float(c)] for a, b, c in zones],
            "where": where_label(s_min),
            "xy": list(map(float, skeleton_xy(s_min))), "pose": pose_min,
            "profile": [[float(s), float(c)] for s, c, _ in sweep]}


def result_summary_lines(result):
    """図に載せる結論の1行サマリー(英語、L.result_summary_linesの折り返し版)。"""
    lines = []
    bn_box = result.get("bottleneck_furniture_only")
    if bn_box:
        lines.append(f"furniture alone: tightest at {bn_box['where']}, "
                     f"{bn_box['capacity_cm']:.1f}cm margin at best pose")
    bn = result.get("bottleneck")
    if bn and bn["capacity_cm"] < 0:
        nc = result["meta"]["carrier"]["num_carriers"]
        tilt = result["meta"]["carrier"]["max_tilt_deg"]
        nz = len(bn.get("blocked_zones", []))
        lines.append(f"with {nc} carriers: {-bn['capacity_cm']:.1f}cm short at "
                     f"{bn['where']} (best pose within {tilt:.0f} deg tilt"
                     + (f", {nz} blocked spots" if nz > 1 else "") + ")")
    return lines


def result_meta(num_carriers, max_iter, seed):
    return {
        "stair": {"shape": "u-turn", "width": STAIR_WIDTH, "wall": WALL,
                  "rise": RISE, "tread": TREAD,
                  "n_steps1": N_STEPS1, "n_steps2": N_STEPS2,
                  "landing": [X_R1, STAIR_WIDTH], "landing_z": LAND_Z,
                  "base_d": BASE_D, "top_d": TOP_D, "ceil_clear": CEIL_CLEAR},
        "furniture": {"L": L.FURN_L, "W": L.FURN_W, "H": L.FURN_H},
        "carrier": {"r": L.p.HUMAN_R, "height": L.p.HUMAN_HEIGHT, "arm": L.p.CARRY_ARM,
                    "leg_clear": L.LEG_CLEAR,
                    "reach": [L.REACH_MIN, L.REACH_MAX], "max_tilt_deg": L.MAX_TILT_DEG,
                    "num_carriers": num_carriers},
        "planner": {"max_iter": max_iter, "seed": seed, "w_rot": L.W_ROT},
    }


def plan(with_human, outer, obstacles, num_carriers, max_iter, seed):
    """1ケース分のRRT-Connectを回し、結果JSONの1項目(dict)を返す。"""
    import time
    t0 = time.time()
    path, tree = L.p.rrt_connect(
        START, GOAL, with_human=with_human, outer=outer, obstacles=obstacles,
        num_carriers=num_carriers, max_iter=max_iter, seed=seed, w_rot=L.W_ROT,
        sampler=make_sampler(with_human), validator=state_valid, return_trees=True)
    dt = time.time() - t0
    if path is not None:
        return {"found": True, "path": L._path_to_json(path), "time_s": dt}
    be = best_effort_path(tree)
    return {"found": False, "time_s": dt,
            "reached_s": skeleton_s(be[-1][0][0], be[-1][0][1]),
            "best_effort_path": L._path_to_json(be)}


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--carriers", type=int, choices=(1, 2), default=L.p.NUM_CARRIERS)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--furniture", type=float, nargs=3, metavar=("L", "W", "H"),
                        help="家具の寸法(長さ 幅 高さ, cm)。既定はL字と同じ200x50x65")
    parser.add_argument("--width", type=float, help="階段・踊り場奥行きの幅(cm)")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="ボトルネック掃引を省く(動作確認用)")
    parser.add_argument("--out", default=os.path.join("results", "ustair_result.json"))
    args = parser.parse_args()
    num_carriers = args.carriers

    if args.furniture:
        L.FURN_L, L.FURN_W, L.FURN_H = sorted(args.furniture, reverse=True)
    configure(width=args.width)
    outer, obstacles = build_ustairs()
    print(f"折り返し階段: 幅{STAIR_WIDTH:.0f}cm, 蹴上げ{RISE:.0f}cm x ({N_STEPS1}+{N_STEPS2})段, "
          f"踊り場{X_R1:.0f}x{STAIR_WIDTH:.0f}cm / "
          f"家具: {L.FURN_L:.0f}x{L.FURN_W:.0f}x{L.FURN_H:.0f}cm / 運搬者: {num_carriers}人")

    result = {"meta": result_meta(num_carriers, args.max_iter, args.seed)}
    for key, with_human, label in (("box_only", False, "家具単体"),
                                   ("with_carriers", True, f"運搬者あり({num_carriers}人)")):
        print(f"\n{label}の経路を探索中...")
        r = plan(with_human, outer, obstacles, num_carriers, args.max_iter, args.seed)
        if r["found"]:
            print(f"  PASS [{r['time_s']:.0f}s]")
        else:
            print(f"  BLOCKED (この試行回数では見つからず; 弧長{r['reached_s']:.0f}"
                  f"/{S_TOTAL:.0f}cmまで到達) [{r['time_s']:.0f}s]")
        result[key] = r

    if not args.skip_sweep:
        import time
        for key, with_human, label in (("bottleneck_furniture_only", False, "家具単体"),
                                       ("bottleneck", True, f"運搬者あり({num_carriers}人)")):
            print(f"\nボトルネック掃引({label}、中心線に沿って)...")
            t0 = time.time()
            bn = bottleneck_entry(sweep_capacity(outer, obstacles, num_carriers,
                                                 with_human=with_human))
            print(f"  最も狭い位置: {bn['where']} (弧長{bn['s']:.0f}cm), "
                  f"最良姿勢での余裕={bn['capacity_cm']:.1f}cm [{time.time() - t0:.0f}s]")
            if bn["capacity_cm"] < 0:
                print(f"  -> この位置では、どの姿勢でもあと{-bn['capacity_cm']:.1f}cm足りない"
                      "(姿勢グリッドの範囲で)")
                for a, b, c in bn["blocked_zones"]:
                    print(f"     詰まる区間: 弧長{a:.0f}..{b:.0f}cm で最大{-c:.1f}cm不足")
            result[key] = bn

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nsaved result to {args.out}")
