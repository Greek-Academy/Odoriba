"""
Phase 2 拡張: 回り階段(上り5段 -> 角で30度ずつ3段回る -> 直交する上り4段)。

L字階段(phase2_lstair.py)の平らな踊り場を、扇形の踏み面3段(30度x3)に
置き換えた形。日本の戸建てで踊り場を取る余裕がないときの定番で、
蹴上げの数はL字と同じ12段。踊り場と違い角でも床が上がっていくため、
回転の途中で前後の運搬者の足元の高さが変わる。

L字・折り返し(phase2_ustair.py)との分担:
  - 家具形状・可搬性オラクル・RRT-Connect は phase2_lstair / phase2_demo を
    そのまま使う(オラクルには歩行面関数 floor_fn だけ差し替えて渡す)
  - このファイルが持つのは幾何だけ: 環境の直方体、歩行面の高さ、中心線、
    骨格誘導サンプラー、ボトルネック掃引

座標系はL字と同じ: 下の廊下を+y方向に進み、角で+x方向に折れる。
角の正方形(幅W x W)の内側の角 PIVOT=(W, FL1_Y1) を中心に、扇形の段を
入口側から 1段目(角度150..180度) / 2段目(120..150度) / 3段目(90..120度)
とする(角度はPIVOTから見た+x軸基準)。

    y
    ^   +--------+------------------------+
    |   | 回り段 |  フライト2(+xへ上る) 上廊下 |
    |   | 3/2/1  |                        |
    |   +------PIVOT----------------------+
    |   |フライト1|
    |   |(+yへ上る)|
    |   | 下廊下  |
    +---+--------+--> x

扇形の段の表し方: 扇形は直方体では書けないので、「その段より上の段を
すべて含む扇形」を、PIVOTを角に置いたz軸まわりに回転した直方体で
下から積む(2段目以降 = 60..150度の直角の範囲、3段目 = 30..120度)。
直角のうち正方形からはみ出す部分は、外枠の外(壁の中)か、
より高いフライト2の段の中に埋まるので判定には影響しない。
ただし3Dビューアでは壁の外にはみ出した灰色の箱として見える。

実行(リポジトリの phase2/ で):
    python phase2_winder.py --max-iter 800 --skip-sweep   # 動作確認用
    python phase2_winder.py                               # 本番(数分かかる)
"""

import numpy as np

import phase2_lstair as L

# ---- パラメータ(cm) ----
# L字と同じ「ごく普通の戸建て」の仮値。蹴上げの合計(12段)もL字と揃える。
STAIR_WIDTH = 90.0    # 廊下・各フライトの幅(=角の正方形の一辺)
RISE = 17.0           # 蹴上げ
TREAD = 26.0          # 踏み面(直進部分)
N_STEPS1 = 5          # 下側フライトの段数
N_WINDERS = 3         # 角の回り段の数(90度をこの数で等分)
N_STEPS2 = 4          # 上側フライトの段数
BASE_D = 340.0        # 下の廊下の奥行き(START用の余白)
TOP_D = 340.0         # 上の廊下の奥行き(GOAL用の余白)
CEIL_CLEAR = 220.0    # 各フライト最上段の踏み面から天井までの高さ


def _derive():
    """寸法パラメータから派生量(座標・中心線)を作り直す。"""
    global FL1_Y0, FL1_Y1, LAND_Y1, PIVOT, WIND_DEG, FL2_X0, FL2_X1, TOP_X1, TOP_Z
    global CENTER, LAND_YC, S_CORNER, S_TOTAL, CORNER_S0, CORNER_S1, STAIR_ANGLE_DEG
    FL1_Y0 = BASE_D                     # フライト1の開始y
    FL1_Y1 = BASE_D + N_STEPS1 * TREAD  # フライト1の終了y = 角の開始y
    LAND_Y1 = FL1_Y1 + STAIR_WIDTH      # 角の終了y(奥の壁)
    PIVOT = np.array([STAIR_WIDTH, FL1_Y1])  # 回り段の中心(L字の内側の角)
    WIND_DEG = 90.0 / N_WINDERS         # 回り段1段あたりの角度
    FL2_X0 = STAIR_WIDTH                # フライト2の開始x
    FL2_X1 = FL2_X0 + N_STEPS2 * TREAD  # フライト2の終了x = 上廊下の開始x
    TOP_X1 = FL2_X1 + TOP_D             # 上廊下の終了x
    TOP_Z = (N_STEPS1 + N_WINDERS + N_STEPS2) * RISE  # 上廊下の床の高さ

    # 中心線はL字と同じ: (CENTER, 0) -> (CENTER, LAND_YC) -> (TOP_X1, LAND_YC)
    CENTER = STAIR_WIDTH / 2
    LAND_YC = FL1_Y1 + STAIR_WIDTH / 2
    S_CORNER = LAND_YC
    S_TOTAL = S_CORNER + (TOP_X1 - CENTER)
    # コーナーゾーン: 角に入る少し手前から、出た少し先まで
    CORNER_S0 = FL1_Y1 - 40.0
    CORNER_S1 = S_CORNER + (STAIR_WIDTH - CENTER) + 40.0
    STAIR_ANGLE_DEG = np.degrees(np.arctan2(RISE, TREAD))


_derive()


def configure(width=None, rise=None, tread=None, n_steps1=None, n_steps2=None,
              base_d=None, top_d=None, ceil_clear=None):
    """回り階段の寸法を数値で設定し、派生量とSTART/GOALを作り直す。
    指定しない引数は現在値を維持する(L.configureと同じ使い方)。"""
    global STAIR_WIDTH, RISE, TREAD, N_STEPS1, N_STEPS2, BASE_D, TOP_D, CEIL_CLEAR
    if width is not None: STAIR_WIDTH = float(width)
    if rise is not None: RISE = float(rise)
    if tread is not None: TREAD = float(tread)
    if n_steps1 is not None: N_STEPS1 = int(n_steps1)
    if n_steps2 is not None: N_STEPS2 = int(n_steps2)
    if base_d is not None: BASE_D = float(base_d)
    if top_d is not None: TOP_D = float(top_d)
    if ceil_clear is not None: CEIL_CLEAR = float(ceil_clear)
    _derive()
    make_start_goal()


def winder_top_z(k):
    """k段目(1..N_WINDERS)の回り段の踏み面の高さ。"""
    return (N_STEPS1 + k) * RISE


# ---- 環境 ----

def _aabb(x0, x1, y0, y1, z0, z1):
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    h = np.array([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2])
    return (c, np.eye(3), h)


def _quadrant_box(theta0_deg, z_top):
    """PIVOTから見て角度 theta0..theta0+90度 の直角の範囲を覆う、
    z軸まわりに回転した直方体(z=0..z_top)。

    直方体の角をPIVOTに置き、辺の一方を theta0 方向(ローカルx)、
    もう一方を theta0+90 方向(ローカルy)に沿わせる。長さは角の正方形を
    覆いきれる最小限(正方形の頂点の射影の最大値)にする。
    """
    t = np.radians(theta0_deg)
    u = np.array([np.cos(t), np.sin(t)])
    v = np.array([-np.sin(t), np.cos(t)])
    w = STAIR_WIDTH
    rel = np.array([[0.0, 0.0], [-w, 0.0], [-w, w], [0.0, w]])  # 正方形の頂点(PIVOT基準)
    eu = max(float(np.max(rel @ u)), 1.0)
    ev = max(float(np.max(rel @ v)), 1.0)
    rot = np.array([[u[0], v[0], 0.0], [u[1], v[1], 0.0], [0.0, 0.0, 1.0]])
    c_xy = PIVOT + u * eu / 2 + v * ev / 2
    return (np.array([c_xy[0], c_xy[1], z_top / 2]), rot,
            np.array([eu / 2, ev / 2, z_top / 2]))


def build_winder():
    """回り階段の環境を返す。

    戻り値: (outer, obstacles)  ※L.build_lstairsと同じ形式
      outer = 直方体2つのリスト(合併が自由空間の輪郭)
        A: 下廊下+フライト1+角(y方向の棟)
        B: 角+フライト2+上廊下(x方向の棟)
      obstacles = ステップ・スラブの直方体リスト(回り段は回転した直方体)
    """
    outer = [
        _aabb(0, STAIR_WIDTH, 0, LAND_Y1, 0, winder_top_z(N_WINDERS) + CEIL_CLEAR),
        _aabb(0, TOP_X1, FL1_Y1, LAND_Y1, 0, TOP_Z + CEIL_CLEAR),
    ]

    obstacles = []
    # フライト1(+y方向へ上る): i段目のy範囲の下を実体で塞ぐ
    for i in range(N_STEPS1):
        y0 = FL1_Y0 + TREAD * i
        obstacles.append(_aabb(0, STAIR_WIDTH, y0, y0 + TREAD, 0, RISE * (i + 1)))
    # 回り段1段目: 角の正方形全体を1段目の高さまで塞ぐ
    obstacles.append(_aabb(0, STAIR_WIDTH, FL1_Y1, LAND_Y1, 0, winder_top_z(1)))
    # 回り段k段目(k>=2): 角度 90..(180 - (k-1)*WIND_DEG) の扇形をk段目の高さまで。
    # 扇形の終わりの辺(180 - (k-1)*WIND_DEG 度)を直角の上側の辺に合わせる
    for k in range(2, N_WINDERS + 1):
        theta_hi = 180.0 - (k - 1) * WIND_DEG
        obstacles.append(_quadrant_box(theta_hi - 90.0, winder_top_z(k)))
    # フライト2(+x方向へ上る)
    z_corner = winder_top_z(N_WINDERS)
    for i in range(N_STEPS2):
        x0 = FL2_X0 + TREAD * i
        obstacles.append(_aabb(x0, x0 + TREAD, FL1_Y1, LAND_Y1, 0, z_corner + RISE * (i + 1)))
    # 上廊下の床下スラブ
    obstacles.append(_aabb(FL2_X1, TOP_X1, FL1_Y1, LAND_Y1, 0, TOP_Z))
    return outer, obstacles


def winder_index(x, y):
    """角の正方形の中の点が何段目の回り段の上か(1..N_WINDERS)。"""
    ang = np.degrees(np.arctan2(y - PIVOT[1], x - PIVOT[0]))
    if ang < 90.0:   # PIVOTちょうど・数値誤差で境界の外に出た点
        ang = 90.0
    k = int((180.0 - ang) // WIND_DEG) + 1
    return min(max(k, 1), N_WINDERS)


def floor_z(x, y, z_hint=None):
    """(x, y) の真下の歩行面の高さ。歩ける場所でなければNone。

    可搬性オラクル(L.state_valid等)には floor_fn=floor_z として渡す。

    z_hint(高さのヒント)はスキャン空間の床(scan_space)と呼び出し方を
    揃えるための引数で、ここでは床が重ならないので使わない。
    """
    if 0.0 <= x <= STAIR_WIDTH and 0.0 <= y <= LAND_Y1:
        if y < FL1_Y0:
            return 0.0
        if y < FL1_Y1:
            i = min(int((y - FL1_Y0) // TREAD), N_STEPS1 - 1)
            return RISE * (i + 1)
        return winder_top_z(winder_index(x, y))
    if FL1_Y1 <= y <= LAND_Y1 and STAIR_WIDTH < x <= TOP_X1:
        if x < FL2_X1:
            i = min(int((x - FL2_X0) // TREAD), N_STEPS2 - 1)
            return winder_top_z(N_WINDERS) + RISE * (i + 1)
        return TOP_Z
    return None


# ---- 骨格(中心線) ----

def skeleton_xy(s):
    """弧長s(0..S_TOTAL)における中心線上の点(x, y)。"""
    if s <= S_CORNER:
        return CENTER, s
    return CENTER + (s - S_CORNER), LAND_YC


def skeleton_heading(s):
    """弧長sでの進行方向のヨー(度)。フライト1=90, フライト2=0。"""
    return 90.0 if s <= S_CORNER else 0.0


def skeleton_s(x, y):
    """(x, y) を中心線に射影したときの弧長s(進捗の測定用)。"""
    t1 = np.clip(y, 0.0, S_CORNER)
    d1 = np.hypot(x - CENTER, y - t1)
    t2 = np.clip(x, CENTER, TOP_X1)
    d2 = np.hypot(x - t2, y - LAND_YC)
    return float(t1 if d1 <= d2 else S_CORNER + (t2 - CENTER))


def on_flight(s):
    """弧長sがフライト(直進の段の上)にあるか。勾配ピッチのヒントに使う。"""
    return (FL1_Y0 <= s <= FL1_Y1) or \
           (S_CORNER + (FL2_X0 - CENTER) <= s <= S_CORNER + (FL2_X1 - CENTER))


def walk_z(s):
    """中心線に沿った弧長sでの歩行面の高さ(サンプリング誘導用)。"""
    z = floor_z(*skeleton_xy(s))
    return 0.0 if z is None else z


def min_center_z(s, pitch_deg):
    """弧長sで長軸をpitchだけ傾けたとき、家具の最下点が歩行面に触れる
    ときの中心高さ(L.min_center_zの回り階段版)。"""
    pr = np.radians(abs(pitch_deg))
    return walk_z(s) + (L.FURN_L / 2) * np.sin(pr) + (L.FURN_H / 2) * np.cos(pr)


def where_label(s):
    """弧長sの場所の英語ラベル(図の注記用)。"""
    return "winder" if FL1_Y1 <= s <= S_CORNER + (STAIR_WIDTH - CENTER) else f"s={s:.0f}cm"


# ---- START / GOAL ----

def make_start_goal():
    """START(下廊下、長軸+y)とGOAL(上廊下、長軸+x)を作り直す。"""
    global START, GOAL
    START = (np.array([CENTER, 170.0, L.FURN_H / 2]), L._pose(90.0, 0.0).as_quat())
    GOAL = (np.array([FL2_X1 + 170.0, LAND_YC, TOP_Z + L.FURN_H / 2]),
            L._pose(0.0, 0.0).as_quat())


make_start_goal()


# ---- 骨格誘導サンプラー / validator ----

def make_sampler(with_human, p_guided=0.8, pos_noise=10.0, rot_noise_deg=8.0):
    """回り階段用の骨格誘導サンプラー(L.make_samplerと同じ構造)。

    L字との違いは高さだけ: 角では床が回り段で上がっていくので、
    誘導する中心高さを回り段の歩行面から取る(L字の平らな踊り場の
    高さのままだと、角のサンプルが段にめり込んで無駄になる)。
    """
    max_pitch = L.MAX_TILT_DEG if with_human else 92.0

    def sampler(rng):
        if rng.random() >= p_guided:
            pos = rng.uniform([0, 0, 0], [TOP_X1, LAND_Y1, TOP_Z + CEIL_CLEAR])
            return pos, L.g3.random_quaternion(rng)
        s = rng.uniform(0.0, S_TOTAL)
        if CORNER_S0 <= s <= CORNER_S1:
            yaw = rng.uniform(-15.0, 105.0)
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
    """rrt_connectのvalidator。L字のオラクルに回り階段の歩行面を渡すだけ。"""
    return L.state_valid(pos, quat, with_human, outer, obstacles, num_carriers,
                         floor_fn=floor_z)


def best_effort_path(tree):
    """RRTが失敗したとき、start側ツリーで中心線上を一番先まで進めたノード
    までの経路(L.best_effort_pathの回り階段版)。"""
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
    L.sweep_capacityの回り階段版(姿勢グリッドの考え方は同じ)。
    """
    max_pitch = L.MAX_TILT_DEG if with_human else 90.0
    pitches = [t for t in (0.0, 15.0, 30.0, 45.0, 55.0, 70.0, 85.0, 90.0) if t <= max_pitch]
    laterals = (-10.0, 0.0, 10.0)
    z_extras = (0.0, 12.0, 30.0)

    s_margin = L.FURN_L / 2 + L.p.CARRY_ARM + L.p.HUMAN_R + 10.0
    results = []
    for s in np.arange(s_margin, S_TOTAL - s_margin + 1e-9, ds):
        x0, y0 = skeleton_xy(s)
        in_corner = CORNER_S0 <= s <= CORNER_S1
        yaws = (0.0, 22.5, 45.0, 67.5, 90.0) if in_corner else (skeleton_heading(s),)
        best = -np.inf
        best_pose = None
        for yaw in yaws:
            for pitch in pitches:
                quat = L._pose(yaw, pitch).as_quat()
                z_base = min_center_z(s, pitch)
                for lat in laterals:
                    # 横方向 = 中心線に直交する向き
                    x = x0 + (lat if s <= S_CORNER else 0.0)
                    y = y0 + (0.0 if s <= S_CORNER else lat)
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
    return {"s": float(s_min), "capacity_cm": float(c_min),
            "where": where_label(s_min),
            "xy": list(map(float, skeleton_xy(s_min))), "pose": pose_min,
            "profile": [[float(s), float(c)] for s, c, _ in sweep]}


def result_summary_lines(result):
    """図に載せる結論の1行サマリー(英語、L.result_summary_linesの回り階段版)。"""
    lines = []
    bn_box = result.get("bottleneck_furniture_only")
    if bn_box:
        lines.append(f"furniture alone: tightest at {bn_box['where']}, "
                     f"{bn_box['capacity_cm']:.1f}cm margin at best pose")
    bn = result.get("bottleneck")
    if bn and bn["capacity_cm"] < 0:
        nc = result["meta"]["carrier"]["num_carriers"]
        tilt = result["meta"]["carrier"]["max_tilt_deg"]
        lines.append(f"with {nc} carriers: {-bn['capacity_cm']:.1f}cm short at "
                     f"{bn['where']} (best pose within {tilt:.0f} deg tilt)")
    return lines


def result_meta(num_carriers, max_iter, seed):
    return {
        "stair": {"shape": "winder", "width": STAIR_WIDTH, "rise": RISE, "tread": TREAD,
                  "n_steps1": N_STEPS1, "n_winders": N_WINDERS, "n_steps2": N_STEPS2,
                  "base_d": BASE_D, "top_d": TOP_D, "ceil_clear": CEIL_CLEAR},
        "furniture": {"L": L.FURN_L, "W": L.FURN_W, "H": L.FURN_H},
        "carrier": {"r": L.p.HUMAN_R, "height": L.p.HUMAN_HEIGHT, "arm": L.p.CARRY_ARM,
                    "leg_clear": L.LEG_CLEAR,
                    "reach": [L.REACH_MIN, L.REACH_MAX], "max_tilt_deg": L.MAX_TILT_DEG,
                    "num_carriers": num_carriers, **L.body_meta()},
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
    parser.add_argument("--width", type=float, help="階段の幅(cm)")
    parser.add_argument("--html", help="3Dビューア(HTML)の出力先。--jsonと併用すると"
                                       "探索せず保存済みの結果から書き出すだけにする")
    parser.add_argument("--json", help="保存済みの結果JSON(--htmlの再生成用)")
    parser.add_argument("--cdn", action="store_true",
                        help="plotly.jsを埋め込まずCDNから読む(ファイルが小さくなる)")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="ボトルネック掃引を省く(動作確認用)")
    parser.add_argument("--out", default=os.path.join("results", "winder_result.json"))
    args = parser.parse_args()
    num_carriers = args.carriers

    if args.furniture:
        L.FURN_L, L.FURN_W, L.FURN_H = sorted(args.furniture, reverse=True)
    configure(width=args.width)
    outer, obstacles = build_winder()
    print(f"回り階段: 幅{STAIR_WIDTH:.0f}cm, 蹴上げ{RISE:.0f}cm x "
          f"({N_STEPS1}+回り{N_WINDERS}+{N_STEPS2})段 / "
          f"家具: {L.FURN_L:.0f}x{L.FURN_W:.0f}x{L.FURN_H:.0f}cm / 運搬者: {num_carriers}人")

    def write_html(result):
        import phase2_lstair_3d as viz3d
        box, car = result["box_only"], result["with_carriers"]
        nc = result["meta"]["carrier"]["num_carriers"]
        fu = result["meta"]["furniture"]
        lines = [f"家具 {fu['L']:.0f}×{fu['W']:.0f}×{fu['H']:.0f}cm / 階段幅 "
                 f"{result['meta']['stair']['width']:.0f}cm  —  ドラッグで回転、Playで再生"]
        bn = result.get("bottleneck")
        if bn and bn["capacity_cm"] < 0:
            lines.append(f"{nc}人で運ぶと、回り段でどう向けてもあと{-bn['capacity_cm']:.0f}cm足りない")
        viz3d.render_html(
            result, args.html, use_cdn=args.cdn, env=(outer, obstacles),
            floor_fn=floor_z, summary_lines=result_summary_lines,
            camera_eye=dict(x=-1.0, y=-0.9, z=1.9),
            case_titles=(f"家具だけ: {'通る' if box['found'] else '通らない'}",
                         f"{nc}人で運ぶ: " + ("通る" if car['found'] else
                                           "通らない(赤=探索が進めた最後の姿勢)")),
            title_text=("回り階段: 家具だけなら通るのに、人が運ぶと角で回せない"
                        "<br><sup>" + "<br>".join(lines) + "</sup>"))

    if args.json:
        # 探索はせず、保存済みの経路を再生するだけ(寸法は結果JSONのものを使う)
        with open(args.json) as f:
            result = json.load(f)
        st, fu = result["meta"]["stair"], result["meta"]["furniture"]
        L.FURN_L, L.FURN_W, L.FURN_H = fu["L"], fu["W"], fu["H"]
        configure(width=st["width"], rise=st["rise"], tread=st["tread"],
                  n_steps1=st["n_steps1"], n_steps2=st["n_steps2"],
                  base_d=st["base_d"], top_d=st["top_d"], ceil_clear=st["ceil_clear"])
        outer, obstacles = build_winder()
        if args.html:
            write_html(result)
        raise SystemExit(0)

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
            if with_human:
                bn["breakdown"] = L.clearance_breakdown(
                    np.array(bn["pose"][0]), np.array(bn["pose"][1]),
                    outer, obstacles, num_carriers, floor_fn=floor_z)
                print("  " + L.format_breakdown(bn["breakdown"]))
            result[key] = bn

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nsaved result to {args.out}")
    if args.html:
        write_html(result)
