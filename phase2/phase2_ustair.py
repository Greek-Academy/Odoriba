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
