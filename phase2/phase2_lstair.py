"""
Phase 2 拡張: L字階段(上り6段 -> 踊り場90x90cm -> 直交する上り6段)。

phase2_demo.py の直線階段を、Odoribaの名前の由来である「踊り場」まで
拡張したもの。見せたい差は Phase 1 と同じ:

    家具単体(自由に浮遊する剛体)なら通るが、
    運搬者を付けると踊り場の回転で詰まる。

このファイルの読み方(上から順に):
  1. パラメータ(cm) -- L字階段・家具・運搬者の寸法
  2. 環境 -- 外枠を「直方体2つの合併」で表し(phase2_demoで一般化済み)、
     ステップ・スラブをくり抜く。歩行面の高さ floor_z(x, y) もここ
  3. 可搬性オラクル(このモジュールでの精密化) -- phase2_demoの
     「胴体は把持点の近くに浮いている」近似をやめ、
       - 運搬者は自分の(x, y)の歩行面の上に立つ(カプセルは床から身長分)
       - 把持点の高さは足元から REACH_MIN..REACH_MAX の範囲(手が届くこと)
       - 家具の傾き(長軸の仰角)は MAX_TILT_DEG まで(静的保持可能性の
         簡略版。CLAUDE.mdのオラクル優先順位2「傾き角の上限」)
     とする。これを入れないと、家具を垂直に立てたときに上側の運搬者が
     宙に浮いたまま「通れる」ことになってしまう(実際に起きた)。
  4. 骨格誘導サンプラー -- L字の中心線に沿って位置を、フライトでは
     勾配ピッチを、コーナー(踊り場)ではヨー一様+立て上げピッチを撒く
  5. ボトルネック掃引 -- 中心線上の各位置で「許される姿勢のどれかで
     何cmの余裕が出せるか」を粗い姿勢グリッドで評価し、最悪の位置と
     不足量(あと何cm)を出す
  6. __main__ -- RRT-Connectを2ケース実行し、結果を results/*.json に保存

実行(リポジトリの phase2/ で):
    python phase2_lstair.py                # 本番(数分かかる)
    python phase2_lstair.py --max-iter 200 --skip-sweep   # 動作確認用
"""

import json
import os
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

import geometry3d as g3
import human_figure as hf
import phase2_demo as p

# SE(3)距離の回転項の重み。phase2_demo.W_ROT=0.4は自身のdocstring
# (「家具の外接円半径くらいのオーダーが目安」= この家具では約98cm)と
# 桁が合っておらず、90度回転してもdist_se3が約0.6にしかならない。
# その結果、edge_valid(刻み幅EDGE_RES=4cm)が90度の回転エッジを
# 3点しかチェックせず、コーナーの回転で壁を抜ける経路を「有効」と
# 誤判定し得る。直線階段(回転がほぼない)では実害が出なかったが、
# 踊り場の90度旋回が主役のこの環境では致命的。以前はここで
# p.W_ROT を上書きしていたが、importの副作用(このモジュールを読む
# だけで直線階段デモの挙動が変わる)を避けるため、rrt_connectの
# w_rot引数として都度渡す。
W_ROT = 60.0

# ---- パラメータ(cm) ----
# 階段は実測前の仮値(建築基準法ぎりぎりではなく、ごく普通の戸建て
# 寸法: 蹴上げ17 / 踏み面26 / 幅90)。
STAIR_WIDTH = 90.0    # 廊下・階段・踊り場の幅
RISE = 17.0           # 蹴上げ
TREAD = 26.0          # 踏み面
N_STEPS1 = 6          # 下側フライトの段数
N_STEPS2 = 6          # 上側フライトの段数
BASE_D = 340.0        # 下の廊下の奥行き(START用の余白)
TOP_D = 340.0         # 上の廊下の奥行き(GOAL用の余白)
CEIL_CLEAR = 220.0    # 各フライト最上段の踏み面から天井までの高さ

# 派生量。座標系: 下の廊下を+y方向に進み、踊り場で+x方向に折れる。
FL1_Y0 = BASE_D                     # フライト1の開始y
FL1_Y1 = BASE_D + N_STEPS1 * TREAD  # フライト1の終了y = 踊り場の開始y
LAND_Y1 = FL1_Y1 + STAIR_WIDTH      # 踊り場の終了y
LAND_Z = N_STEPS1 * RISE            # 踊り場の床の高さ
FL2_X0 = STAIR_WIDTH                # フライト2の開始x(踊り場の右端)
FL2_X1 = FL2_X0 + N_STEPS2 * TREAD  # フライト2の終了x = 上廊下の開始x
TOP_X1 = FL2_X1 + TOP_D             # 上廊下の終了x
TOP_Z = LAND_Z + N_STEPS2 * RISE    # 上廊下の床の高さ

# 家具はこのモジュールでローカルに定義する(phase2_demoのソファ180x60x80
# ではなく、背の高いタンス/本棚を想定した200x50x65)。理由:
#  - 断面(短辺x高さ)の対角線が階段幅より小さくないと、垂直に立てても
#    踊り場で回せず、「家具単体なら通る」側が成立しない。
#      60x80の断面対角 = 100cm > 階段幅90cm -> 家具単体でも回らない
#      50x65の断面対角 =  82cm < 90cm       -> 立てれば回る(余裕4cm/側)
#  - 長辺は、傾き上限内の運搬者ありが踊り場を回れない長さにする。
#    180cmでは傾き約50度の対角ピボット(鼻先を上フライトの吹き抜けに
#    突っ込みつつ回る)で運搬者ありでも通ってしまうことがRRTで見つかった。
#    200cmではその隙間が閉じる(ボトルネック掃引の不足量が裏付け)。
FURN_L = 200.0   # 長辺
FURN_W = 50.0    # 短辺
FURN_H = 65.0    # 高さ
# 運搬者(HUMAN_R, HUMAN_HEIGHT, CARRY_ARM, SIDE_OFFSETS)はphase2_demoの
# 値を使い回す。

# このモジュールで足す保持拘束(いずれも実測なしの常識的な仮値):
REACH_MIN = 10.0      # 把持点は足元からこれ以上の高さ(床すれすれは持てない)
REACH_MAX = 190.0     # 把持点は足元からこれ以下の高さ(手を上げて届く上限)
MAX_TILT_DEG = 55.0   # 運搬時の長軸の仰角の上限(静的保持可能性の簡略版)
# 運搬者カプセルの下端は足元の床からこの高さより上だけを占有として扱う。
# 床から全身を円柱にすると、階段では下端が必ず隣の一段高い段(蹴上げ17cm)
# に食い込んで「人は階段に立てない」ことになってしまう。下腿は細く
# 隣の段と干渉しない、という近似。
LEG_CLEAR = 35.0


def build_lstairs():
    """L字階段の環境を返す。

    戻り値: (outer, obstacles)
      outer = 直方体2つのリスト(合併が自由空間の輪郭)
        A: 下廊下+フライト1+踊り場(y方向の棟)
        B: 踊り場+フライト2+上廊下(x方向の棟)
      obstacles = ステップ・スラブの直方体リスト
    """
    def aabb(x0, x1, y0, y1, z0, z1):
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        h = np.array([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2])
        return (c, np.eye(3), h)

    outer = [
        aabb(0, STAIR_WIDTH, 0, LAND_Y1, 0, LAND_Z + CEIL_CLEAR),
        aabb(0, TOP_X1, FL1_Y1, LAND_Y1, 0, TOP_Z + CEIL_CLEAR),
    ]

    obstacles = []
    # フライト1: i段目のy範囲の下(z=0から段の高さまで)を実体で塞ぐ
    for i in range(N_STEPS1):
        y0 = FL1_Y0 + TREAD * i
        obstacles.append(aabb(0, STAIR_WIDTH, y0, y0 + TREAD, 0, RISE * (i + 1)))
    # 踊り場の床下スラブ
    obstacles.append(aabb(0, STAIR_WIDTH, FL1_Y1, LAND_Y1, 0, LAND_Z))
    # フライト2(+x方向へ上る)
    for i in range(N_STEPS2):
        x0 = FL2_X0 + TREAD * i
        obstacles.append(aabb(x0, x0 + TREAD, FL1_Y1, LAND_Y1, 0, LAND_Z + RISE * (i + 1)))
    # 上廊下の床下スラブ
    obstacles.append(aabb(FL2_X1, TOP_X1, FL1_Y1, LAND_Y1, 0, TOP_Z))
    return outer, obstacles


def floor_z(x, y, z_hint=None):
    """(x, y) の真下の歩行面の高さ。歩ける場所でなければNone。

    運搬者オラクル(足元の床)とサンプラー(誘導高さ)の両方で使う。

    z_hint(高さのヒント)はスキャン空間の床(scan_space)と呼び出し方を
    揃えるための引数で、ここでは床が重ならないので使わない。
    """
    if 0.0 <= x <= STAIR_WIDTH and 0.0 <= y <= LAND_Y1:
        if y < FL1_Y0:
            return 0.0
        if y < FL1_Y1:
            i = min(int((y - FL1_Y0) // TREAD), N_STEPS1 - 1)
            return RISE * (i + 1)
        return LAND_Z
    if FL1_Y1 <= y <= LAND_Y1 and 0.0 <= x <= TOP_X1:
        if x < FL2_X0:
            return LAND_Z
        if x < FL2_X1:
            i = min(int((x - FL2_X0) // TREAD), N_STEPS2 - 1)
            return LAND_Z + RISE * (i + 1)
        return TOP_Z
    return None


# ---- 骨格(中心線) ----
# L字のポリライン: (45, 0) -> (45, 踊り場中心y) -> (上廊下の端, 踊り場中心y)
CENTER = STAIR_WIDTH / 2
LAND_YC = FL1_Y1 + STAIR_WIDTH / 2   # 踊り場の中心y
S_CORNER = LAND_YC                    # コーナー点の弧長(セグメント1の長さ)
S_TOTAL = S_CORNER + (TOP_X1 - CENTER)

# コーナー(踊り場)ゾーン: この弧長範囲では回転ヒント(ヨー一様+
# 立て上げピッチ)を撒く。踊り場の前後に少し広げてある。
CORNER_S0 = FL1_Y1 - 40.0
CORNER_S1 = S_CORNER + (STAIR_WIDTH - CENTER) + 40.0


def configure(width=None, rise=None, tread=None, n_steps1=None, n_steps2=None,
              base_d=None, top_d=None, ceil_clear=None):
    """L字階段の寸法を数値で設定し、派生量(座標・中心線など)を作り直す。

    「実測や好みの数値から仮想階段を組む」ための入口。指定しない引数は
    現在値を維持する。build_lstairs/make_sampler/オラクル等はここで
    再計算したモジュール変数を参照するので、この呼び出し後の判定・可視化は
    新しい寸法で動く。
    """
    global STAIR_WIDTH, RISE, TREAD, N_STEPS1, N_STEPS2, BASE_D, TOP_D, CEIL_CLEAR
    global FL1_Y0, FL1_Y1, LAND_Y1, LAND_Z, FL2_X0, FL2_X1, TOP_X1, TOP_Z
    global CENTER, LAND_YC, S_CORNER, S_TOTAL, CORNER_S0, CORNER_S1, STAIR_ANGLE_DEG
    if width is not None: STAIR_WIDTH = float(width)
    if rise is not None: RISE = float(rise)
    if tread is not None: TREAD = float(tread)
    if n_steps1 is not None: N_STEPS1 = int(n_steps1)
    if n_steps2 is not None: N_STEPS2 = int(n_steps2)
    if base_d is not None: BASE_D = float(base_d)
    if top_d is not None: TOP_D = float(top_d)
    if ceil_clear is not None: CEIL_CLEAR = float(ceil_clear)

    FL1_Y0 = BASE_D
    FL1_Y1 = BASE_D + N_STEPS1 * TREAD
    LAND_Y1 = FL1_Y1 + STAIR_WIDTH
    LAND_Z = N_STEPS1 * RISE
    FL2_X0 = STAIR_WIDTH
    FL2_X1 = FL2_X0 + N_STEPS2 * TREAD
    TOP_X1 = FL2_X1 + TOP_D
    TOP_Z = LAND_Z + N_STEPS2 * RISE
    CENTER = STAIR_WIDTH / 2
    LAND_YC = FL1_Y1 + STAIR_WIDTH / 2
    S_CORNER = LAND_YC
    S_TOTAL = S_CORNER + (TOP_X1 - CENTER)
    CORNER_S0 = FL1_Y1 - 40.0
    CORNER_S1 = S_CORNER + (STAIR_WIDTH - CENTER) + 40.0
    STAIR_ANGLE_DEG = np.degrees(np.arctan2(RISE, TREAD))

    # START/GOALも寸法依存なので作り直す(find_max_*が参照するため)
    global START, GOAL
    START = (np.array([CENTER, 170.0, FURN_H / 2]), _pose(90.0, 0.0).as_quat())
    GOAL = (np.array([FL2_X1 + 170.0, LAND_YC, TOP_Z + FURN_H / 2]),
            _pose(0.0, 0.0).as_quat())


def skeleton_xy(s):
    """弧長s(0..S_TOTAL)における中心線上の点(x, y)。"""
    if s <= S_CORNER:
        return CENTER, s
    return CENTER + (s - S_CORNER), LAND_YC


def skeleton_s(x, y):
    """(x, y) を中心線に射影したときの弧長s(可視化・進捗の測定用)。"""
    # セグメント1 (x=CENTER, y: 0..S_CORNER) までの距離と射影
    t1 = np.clip(y, 0.0, S_CORNER)
    d1 = np.hypot(x - CENTER, y - t1)
    # セグメント2 (y=LAND_YC, x: CENTER..TOP_X1)
    t2 = np.clip(x, CENTER, TOP_X1)
    d2 = np.hypot(x - t2, y - LAND_YC)
    return t1 if d1 <= d2 else S_CORNER + (t2 - CENTER)


def walk_z(s):
    """中心線に沿った弧長sでの歩行面の高さ(サンプリング誘導用)。"""
    z = floor_z(*skeleton_xy(s))
    return 0.0 if z is None else z


STAIR_ANGLE_DEG = np.degrees(np.arctan2(RISE, TREAD))


# ---- 家具・運搬者の形状(このモジュールの寸法で) ----

def furniture_points(pos, quat):
    """家具の表面サンプル点(ワールド座標)。phase2_demoと同じ考え方で
    寸法だけこのモジュールのものを使う。"""
    half = np.array([FURN_L / 2, FURN_W / 2, FURN_H / 2])
    local = g3.box_surface_sample_points(half, nu=6, nv=3, nw=3)
    R = g3.rotmat_from_quat(quat)
    return (local @ R.T) + np.asarray(pos)


def env_clearance_points(pts, outer, obstacles):
    """環境に対する各点の符号付き余裕(cm)の配列。

    outer が clearance_points を持つ環境(スキャン空間 scan_space.ScanSpace)なら
    それで測り、そうでなければ直方体の環境(外枠の合併 - 障害物)として測る。
    オラクル・掃引・RRTは outer/obstacles をここに渡すだけなので、スキャン
    空間を outer に渡せば(obstacles は None)そのまま同じ判定が動く。
    """
    if hasattr(outer, "clearance_points"):
        return outer.clearance_points(pts)
    return p.clearance_points_3d(pts, outer, obstacles)


def furniture_clearance(pos, quat, outer, obstacles):
    """この姿勢での家具単体(運搬者なし)のクリアランス(cm)。"""
    return float(np.min(env_clearance_points(furniture_points(pos, quat), outer, obstacles)))


# 運搬者の胴体円柱: 膝下(LEG_CLEAR)から頭頂まで。ローカル点は使い回す。
_CARRIER_HALF_H = (p.HUMAN_HEIGHT - LEG_CLEAR) / 2
_CARRIER_LOCAL = g3.cylinder_surface_sample_points(p.HUMAN_R, _CARRIER_HALF_H,
                                                   n_theta=10, n_h=3)


def carrier_points(center):
    """運搬者カプセル(LEG_CLEAR..身長の円柱)の表面サンプル点。"""
    return _CARRIER_LOCAL + np.asarray(center)


# 運搬者の体の判定モデル。"humanoid"(既定)は human_figure の人型
# (横幅=肩幅40cm、前後=胴の厚み24cm、腕・頭・膝上の脚)、"cylinder" は
# 従来の円柱(carrier_points)。感度分析(sensitivity.py)で新旧を比べるために残す。
BODY_MODEL = "humanoid"
# 両手の間隔の半分(家具の端面上、運搬者の横位置を中心に左右へ)
HAND_HALF_SPREAD = 18.0


def _carrier_frame(pos, R, sign, side_offset, center):
    """運搬者1人の足元の床の高さ・正面の向き(水平xy)・両手の目標点。

    手の目標点は家具の端面上(把持点の高さ)に置く。正面は家具の中心の
    方向。家具が縦に立っていて真上/真下にあるときは、長軸の水平成分
    (の逆向き)で代用する。判定(carrier_body_clearance)と描画
    (carrier_poses_lstair)で共通に使う。
    """
    hl, hw = FURN_L / 2, FURN_W / 2
    hands = [pos + R @ np.array([sign * hl, float(np.clip(side_offset + d, -hw, hw)), 0.0])
             for d in (HAND_HALF_SPREAD, -HAND_HALF_SPREAD)]
    facing = (pos - center)[:2]
    if np.linalg.norm(facing) < 1e-6:
        facing = -sign * (R @ np.array([1.0, 0.0, 0.0]))[:2]
    if np.linalg.norm(facing) < 1e-6:
        facing = np.array([1.0, 0.0])
    floor = float(center[2] - _CARRIER_HALF_H - LEG_CLEAR)
    return floor, facing, hands


def body_meta():
    """結果JSONのmeta.carrierに足す、体の判定モデルの記録。"""
    return {"body_model": BODY_MODEL, "shoulder_width": 2 * hf.SHOULDER_HW,
            "torso_depth": 2 * hf.TORSO_HD}


def _carrier_signs(n):
    """place_humans_lstairの戻り値の順に対応する、長軸方向の符号。"""
    return (-1.0,) if n == 1 else (1.0, -1.0)


def carrier_body_clearance(pos, quat, side_offset, placed, clearance_fn):
    """運搬者全員の体の余裕(cm)の最小値。

    placedはplace_humans_lstairの戻り値。clearance_fnは点群の各点の
    符号付き距離の配列を返す関数で、直方体の環境なら p.clearance_points_3d、
    メッシュ環境ならその版を渡す(体の形の扱いを環境ごとに書かずに済む)。
    """
    if BODY_MODEL == "cylinder":
        return min(float(np.min(clearance_fn(carrier_points(c)))) for c, _ in placed)
    pos = np.asarray(pos)
    R = g3.rotmat_from_quat(quat)
    parts = []
    for sign, (center, _) in zip(_carrier_signs(len(placed)), placed):
        floor, facing, hands = _carrier_frame(pos, R, sign, side_offset, center)
        parts += hf.body_parts(center[:2], floor, facing, hands, LEG_CLEAR)
    return hf.body_clearance(parts, clearance_fn)


def carrier_furniture_clearance(pos, quat, side_offset, placed):
    """運搬者全員の体と家具自身の余裕(cm)の最小値。負なら体が家具に食い込む。

    把持点は家具の端からCARRY_ARMだけ長軸方向に離した位置で固定なので、
    家具を傾けると端の角が運搬者の頭上・胸元にかぶさり、体が家具の中に
    入った「ありえない姿勢」が成立してしまっていた(壁との余裕しか見て
    いなかったため)。腕は家具を持つために触れているので数えない。
    家具は直方体(FURN_L x W x H)として測る。
    """
    pos = np.asarray(pos)
    R = g3.rotmat_from_quat(quat)
    half = np.array([FURN_L / 2, FURN_W / 2, FURN_H / 2])
    if BODY_MODEL == "cylinder":
        pts = np.vstack([carrier_points(c) for c, _ in placed])
        return float(np.min(g3.obb_sdf_points(pts, pos, R, half)))
    best = np.inf
    for sign, (center, _) in zip(_carrier_signs(len(placed)), placed):
        floor, facing, hands = _carrier_frame(pos, R, sign, side_offset, center)
        for pts, r in hf.body_parts(center[:2], floor, facing, hands, LEG_CLEAR,
                                    with_arms=False):
            best = min(best, float(np.min(g3.obb_sdf_points(pts, pos, R, half))) - r)
    return best


def _pose(yaw_deg, pitch_deg, roll_deg=0.0):
    """ヨー(z)->ピッチ(長軸の仰角)->ロールの順に合成した姿勢。

    長軸はローカルx。長軸を持ち上げるのはローカルyまわりのRy(-pitch)
    (Rx(pitch)は長軸まわりのロールにしかならない -- phase2_demoの
    sample_guidedの修正と同じ話)。
    """
    return (Rotation.from_euler("z", yaw_deg, degrees=True)
            * Rotation.from_euler("y", -pitch_deg, degrees=True)
            * Rotation.from_euler("x", roll_deg, degrees=True))


def tilt_deg(quat):
    """家具の長軸(ローカルx)の水平からの仰角(度、絶対値)。"""
    long_axis = g3.rotmat_from_quat(quat) @ np.array([1.0, 0.0, 0.0])
    return float(np.degrees(np.arcsin(np.clip(abs(long_axis[2]), 0.0, 1.0))))


def min_center_z(s, pitch_deg):
    """弧長sで長軸をpitchだけ傾けたとき、家具の最下点が歩行面に触れる
    ときの中心高さ(誘導サンプリングの下限として使う近似)。"""
    pr = np.radians(abs(pitch_deg))
    return walk_z(s) + (FURN_L / 2) * np.sin(pr) + (FURN_H / 2) * np.cos(pr)


def make_sampler(with_human, p_guided=0.8, pos_noise=10.0, rot_noise_deg=8.0):
    """L字階段用の骨格誘導サンプラーを返す。

    - 位置: 中心線上の弧長sを一様に引き、横方向・高さにノイズ
    - 姿勢: フライト上では長軸を進行方向に向けて勾配ピッチ、
      コーナー(踊り場)ゾーンではヨーを一様(セグメント1の90度から
      セグメント2の0度まで回りきれるように)、ピッチも一様
      (家具単体は85度まで「立てて回す」解を探せるように。運搬者あり
      では傾き上限を超えるサンプルは無駄になるだけなので上限で切る)
    - 確率(1 - p_guided)では完全ランダム(誘導が思いつかない解への保険)
    """
    max_pitch = MAX_TILT_DEG if with_human else 92.0

    def sampler(rng):
        if rng.random() >= p_guided:
            pos = rng.uniform([0, 0, 0], [TOP_X1, LAND_Y1, TOP_Z + CEIL_CLEAR])
            return pos, g3.random_quaternion(rng)
        s = rng.uniform(0.0, S_TOTAL)
        in_corner = CORNER_S0 <= s <= CORNER_S1
        if in_corner:
            yaw = rng.uniform(-15.0, 105.0)
            if with_human:
                pitch = rng.uniform(0.0, max_pitch)
            else:
                # 家具単体の想定解は「垂直近くまで立てて回す」。断面対角が
                # 階段幅を下回るのはピッチ約88度以上だけなので、コーナーでは
                # 垂直付近(55..92度)を重点的に、残りを低い傾きに撒く。
                pitch = rng.uniform(55.0, 92.0) if rng.random() < 0.6 \
                    else rng.uniform(0.0, 55.0)
        else:
            on_flight = (FL1_Y0 <= s <= FL1_Y1) or \
                        (S_CORNER + (FL2_X0 - CENTER) <= s <= S_CORNER + (FL2_X1 - CENTER))
            base_yaw = 90.0 if s <= S_CORNER else 0.0
            base_pitch = STAIR_ANGLE_DEG if on_flight else 0.0
            yaw = base_yaw + rng.normal(0.0, rot_noise_deg)
            pitch = base_pitch + rng.normal(0.0, rot_noise_deg)
        roll = rng.normal(0.0, rot_noise_deg * 0.5)
        quat = _pose(yaw, pitch, roll).as_quat()

        x, y = skeleton_xy(s)
        z = min_center_z(s, pitch) + abs(rng.normal(0.0, 12.0))
        pos = np.array([x, y, z]) + rng.normal(0.0, pos_noise, size=3) * np.array([1, 1, 0.3])
        return pos, quat

    return sampler


# ---- 可搬性オラクル(このモジュールでの精密化) ----

def place_humans_lstair(pos, quat, side_offset, num_carriers=None, floor_fn=None):
    """把持点と運搬者カプセル中心を返す。立てなければNone。

    phase2_demo.place_humans_3dとの違い:
      - 胴体は把持点の近くに「浮かせる」のではなく、把持点の(x, y)の
        歩行面の上に立たせる(カプセルは床から身長分)
      - 把持点が足元からREACH_MIN..REACH_MAXの高さになければ持てない
        (そのときはNoneではなく、余裕をcmで返すために呼び出し側で扱う)

    戻り値: [(capsule_center, reach_margin_cm), ...] または
    None(把持点の真下に歩ける床がない = そこに人は立てない)。
    reach_margin_cmは手の届く範囲までの余裕(負なら届かない)。
    floor_fnは歩行面の高さ関数(既定はこのモジュールのL字のfloor_z)。
    折り返し階段など別の環境でオラクルを使い回すときに差し替える。
    """
    num_carriers = p.NUM_CARRIERS if num_carriers is None else num_carriers
    floor_fn = floor_z if floor_fn is None else floor_fn
    R = g3.rotmat_from_quat(quat)
    pos = np.asarray(pos)
    hl = FURN_L / 2

    out = []
    signs = (-1.0,) if num_carriers == 1 else (1.0, -1.0)
    for sign in signs:
        grip = pos + R @ np.array([sign * (hl + p.CARRY_ARM), side_offset, 0.0])
        # 把持点の高さをヒントに渡す(床が上下に重なる所で正しい段を選ぶため)
        fz = floor_fn(grip[0], grip[1], grip[2])
        if fz is None:
            return None
        hand = grip[2] - fz
        reach_margin = min(hand - REACH_MIN, REACH_MAX - hand)
        center = np.array([grip[0], grip[1], fz + LEG_CLEAR + _CARRIER_HALF_H])
        out.append((center, reach_margin))
    return out


def carriable_clearance_lstair(pos, quat, outer, obstacles, num_carriers=None,
                               floor_fn=None):
    """運搬者込みで達成できる最良のクリアランス(cm)と、その横位置。

    phase2_demo.carriable_clearanceと同じ総当たりだが、
      - 傾き上限を超えていれば即negative(超過角に比例したペナルティを
        cm換算せず、-infで返す。「あと何cm」の対象は幾何の余裕だけ)
      - 手の届く範囲までの余裕(cm)もクリアランスの一項として混ぜる
        (届かない場合は負のcmとして「あとどれだけ低ければ持てたか」になる)

    家具が壁に当たっていても運搬者の項まで評価する。以前は家具の余裕が
    負ならその値をすぐ返していたため、家具が壁にちょうど触れている姿勢
    (余裕 -0.0cm)が、運搬者の項では何十cmも足りなくても「ほぼ0cm」と
    評価され、掃引の最良姿勢に選ばれていた。通る/通らないだけを知りたい
    RRTの判定(state_valid)は、家具の余裕が負なら先に打ち切る。
    """
    if with_tilt_violation(quat):
        return -np.inf, None
    bc = furniture_clearance(pos, quat, outer, obstacles)
    best = -np.inf
    best_offset = None
    for off in p.SIDE_OFFSETS:
        placed = place_humans_lstair(pos, quat, off, num_carriers, floor_fn)
        if placed is None:
            continue
        terms = [bc]
        if ORACLE_TERMS["reach"]:
            terms += [reach_margin for _, reach_margin in placed]
        if ORACLE_TERMS["body"]:
            terms.append(carrier_body_clearance(
                pos, quat, off, placed,
                lambda pts: env_clearance_points(pts, outer, obstacles)))
        if ORACLE_TERMS["self"]:
            terms.append(carrier_furniture_clearance(pos, quat, off, placed))
        cand = min(terms)
        if cand > best:
            best = cand
            best_offset = off
    return best, best_offset


def with_tilt_violation(quat):
    return tilt_deg(quat) > MAX_TILT_DEG


# 可搬性オラクルの項を個別に外すスイッチ。通常は全部True のまま使う。
# 「どの仮定を外すと不足量が何cm変わるか」を測る感度分析(sensitivity.py)専用。
# 傾き上限は項ではなく MAX_TILT_DEG の値そのものを変えて調べる。
# "self" は運搬者の体と家具自身の干渉(carrier_furniture_clearance)。
ORACLE_TERMS = {"reach": True, "body": True, "self": True}

# clearance_breakdownの項の日本語名(出力用)
BREAKDOWN_LABELS = {
    "furniture": "家具が壁・段に当たる",
    "carrier_body": "運搬者の体が壁・段に当たる",
    "carrier_self": "運搬者の体が家具に食い込む",
    "reach": "手が届かない(持つ高さ)",
    "no_floor": "運搬者の立つ床がない",
    "tilt": "傾き上限を超える",
}


def clearance_breakdown(pos, quat, outer, obstacles, num_carriers=None, floor_fn=None):
    """ある姿勢での可搬性オラクルの各項(cm)を名前付きで返す(説明・分析用)。

    carriable_clearance_lstairは最小値しか返さず、家具が壁に当たった時点で
    運搬者の項を評価しないため「なぜ通らないか」が分からない。ここでは
    全項を評価して並べ、一番厳しい項の名前を limiting に入れる。
    運搬者の項は、体と手の届く範囲の両方を含めて最良になる横位置で測る。
    """
    bd = {"furniture": float(furniture_clearance(pos, quat, outer, obstacles)),
          "carrier_body": None, "carrier_self": None, "reach": None,
          "tilt_deg": tilt_deg(quat), "max_tilt_deg": float(MAX_TILT_DEG)}
    best = -np.inf
    for off in p.SIDE_OFFSETS:
        placed = place_humans_lstair(pos, quat, off, num_carriers, floor_fn)
        if placed is None:
            continue
        body = carrier_body_clearance(pos, quat, off, placed,
                                      lambda pts: env_clearance_points(pts, outer, obstacles))
        selfc = carrier_furniture_clearance(pos, quat, off, placed)
        reach = min(m for _, m in placed)
        # 横位置は、オラクルと同じく有効な項だけの最小値が最良になるものを選ぶ
        score = min([v for v, on in ((body, ORACLE_TERMS["body"]),
                                     (selfc, ORACLE_TERMS["self"]),
                                     (reach, ORACLE_TERMS["reach"])) if on],
                    default=np.inf)
        if score > best:
            best = score
            bd["carrier_body"], bd["carrier_self"] = float(body), float(selfc)
            bd["reach"] = float(reach)
    if bd["tilt_deg"] > MAX_TILT_DEG + 1e-6:
        bd["limiting"] = "tilt"
    elif bd["carrier_body"] is None:
        bd["limiting"] = "no_floor" if bd["furniture"] >= 0 else "furniture"
    else:
        # 感度分析で外した項(ORACLE_TERMS が False)は決め手の候補にしない
        enabled = {"carrier_body": ORACLE_TERMS["body"],
                   "carrier_self": ORACLE_TERMS["self"], "reach": ORACLE_TERMS["reach"]}
        bd["limiting"] = min(["furniture"] + [k for k, on in enabled.items() if on],
                             key=lambda k: bd[k])
    return bd


def format_breakdown(bd):
    """clearance_breakdownの結果を1行の日本語にする。"""
    def cm(v):
        return "-" if v is None else f"{v:+.1f}cm"
    line = (f"内訳: 家具{cm(bd['furniture'])} / 運搬者の体{cm(bd['carrier_body'])} / "
            f"体と家具{cm(bd.get('carrier_self'))} / "
            f"手の届く範囲{cm(bd['reach'])} / 傾き{bd['tilt_deg']:.0f}度"
            f"(上限{bd['max_tilt_deg']:.0f}) -> 決め手: {BREAKDOWN_LABELS[bd['limiting']]}")
    if bd["max_tilt_deg"] - bd["tilt_deg"] < 0.5:
        line += "(傾きが上限に張り付いている)"
    return line


def state_valid(pos, quat, with_human, outer, obstacles, num_carriers=None,
                floor_fn=None):
    """このモジュール版のcollision_free(rrt_connectのvalidatorに渡す)。"""
    if furniture_clearance(pos, quat, outer, obstacles) < 0:
        return False
    if with_human:
        cc, _ = carriable_clearance_lstair(pos, quat, outer, obstacles, num_carriers,
                                           floor_fn)
        return cc >= 0
    return True


def carrier_poses_lstair(pos, quat, num_carriers=None, outer=None, obstacles=None,
                         floor_fn=None):
    """描画専用: 最良の横位置で、運搬者を人の形で描くための姿勢のリスト。

    1人ごとに dict(center=判定に使う円柱の中心, foot_xy=足元のxy,
    floor_z=足元の床の高さ, facing=正面の向き(水平xy), hands=両手の目標点2つ)
    を返す。手の目標点は家具の端面上(把持点の高さ)に置く。

    詰まった状態(どの横位置でも不成立)でも「そこに立とうとしている人」を
    描きたいので、成立しない場合はside_offset=0の位置で代用する。
    """
    _, off = carriable_clearance_lstair(pos, quat, outer, obstacles, num_carriers,
                                        floor_fn)
    if off is None:
        off = 0.0
    pos = np.asarray(pos)
    R = g3.rotmat_from_quat(quat)
    signs = _carrier_signs(1 if (num_carriers or p.NUM_CARRIERS) == 1 else 2)
    placed = place_humans_lstair(pos, quat, off, num_carriers, floor_fn)
    if placed is None:
        # 床がない場所: 把持点の高さに浮かせて描く(見た目のためだけ)
        centers = [pos + R @ np.array([sg * (FURN_L / 2 + p.CARRY_ARM), 0.0, 0.0])
                   for sg in signs]
    else:
        centers = [c for c, _ in placed]

    poses = []
    for sg, center in zip(signs, centers):
        floor, facing, hands = _carrier_frame(pos, R, sg, off, center)
        poses.append(dict(center=center, foot_xy=center[:2], floor_z=floor,
                          facing=facing, hands=hands))
    return poses


def best_human_positions_lstair(pos, quat, num_carriers=None, outer=None, obstacles=None,
                                floor_fn=None):
    """描画専用: 最良の横位置での運搬者カプセル中心のリスト。

    詰まった状態(どの横位置でも不成立)でも「そこに立とうとしている人」を
    描きたいので、成立しない場合はside_offset=0の位置で代用する。
    """
    return [cp["center"] for cp in carrier_poses_lstair(pos, quat, num_carriers, outer,
                                                        obstacles, floor_fn)]


# ---- ボトルネック掃引 ----

def sweep_capacity(outer, obstacles, num_carriers=None, ds=15.0,
                   with_human=True, max_pitch=None, s_values=None, fine=False):
    """中心線上の各弧長sについて、粗い姿勢グリッドの中で達成できる
    最良のクリアランス(cm)を返す: [(s, capacity_cm, best_pose), ...]。

    capacityが負の位置は「サンプルした姿勢のどれをとっても
    |capacity| cm足りない」ことを意味する(姿勢グリッドの解像度の
    範囲で。Phase 1のresolution-completeと同じ但し書き)。
    ボトルネック = capacityが最小の位置。

    s_values を渡すとその弧長だけを掃引する。fine=True なら、一番狭い所の
    近くを細かく掃引し直して姿勢も探し直す(phase2_lstair.fine_sweep)。
    """
    if max_pitch is None:
        max_pitch = MAX_TILT_DEG if with_human else 90.0
    if fine:
        def clearance(pos, quat):
            if tilt_deg(quat) > max_pitch + 1e-6:
                return -np.inf
            if with_human:
                return carriable_clearance_lstair(np.asarray(pos), quat, outer, obstacles,
                                                  num_carriers)[0]
            return furniture_clearance(pos, quat, outer, obstacles)
        return fine_sweep(
            lambda sv: sweep_capacity(outer, obstacles, num_carriers, ds, with_human,
                                      max_pitch, s_values=sv),
            clearance, skeleton_heading, ds)
    pitches = [t for t in (0.0, 15.0, 30.0, 45.0, 55.0, 70.0, 85.0, 90.0) if t <= max_pitch]
    laterals = (-10.0, 0.0, 10.0)
    z_extras = (0.0, 12.0, 30.0)

    # 端に近すぎるsは、家具+運搬者の列がモデル化した廊下の端(実際の壁
    # ではなく単に領域の境界)からはみ出すだけなので掃引しない
    # (phase1のEND_MARGINと同じ考え方)。
    s_margin = FURN_L / 2 + p.CARRY_ARM + p.HUMAN_R + 10.0
    results = []
    if s_values is None:
        s_values = np.arange(s_margin, S_TOTAL - s_margin + 1e-9, ds)
    for s in s_values:
        x0, y0 = skeleton_xy(s)
        in_corner = CORNER_S0 <= s <= CORNER_S1
        yaws = (0.0, 22.5, 45.0, 67.5, 90.0) if in_corner else \
               ((90.0,) if s <= S_CORNER else (0.0,))
        best = -np.inf
        best_pose = None
        for yaw in yaws:
            for pitch in pitches:
                quat = _pose(yaw, pitch).as_quat()
                z_base = min_center_z(s, pitch)
                for lat in laterals:
                    # 横方向 = 中心線に直交する向き
                    x = x0 + (lat if s <= S_CORNER else 0.0)
                    y = y0 + (0.0 if s <= S_CORNER else lat)
                    for dz in z_extras:
                        pos = np.array([x, y, z_base + dz])
                        if with_human:
                            c, _ = carriable_clearance_lstair(pos, quat, outer, obstacles,
                                                              num_carriers)
                        else:
                            c = furniture_clearance(pos, quat, outer, obstacles)
                        if c > best:
                            best = c
                            best_pose = (pos.tolist(), quat.tolist())
        results.append((float(s), float(best), best_pose))
    return results


def refine_pose(pos, quat, clearance_fn, heading_deg, step_cm=5.0, step_deg=5.0,
                min_step=0.5, max_evals=300):
    """掃引の粗い姿勢グリッドで見つけた最良姿勢を、その近くで細かく探し直す。

    掃引は横方向±10cm・ピッチ15度刻みなどの粗いグリッドなので、一番狭い所の
    「あと何cm」がグリッドの当たり外れで数cm揺れる。ここでは座標ごとに
    ±刻みを試して良くなれば動き、良くならなければ刻みを半分にする
    (パターン探索)。動かすのは横方向(中心線に直交)・高さ・ヨー・ピッチ・
    ロールで、中心線に沿った方向には動かさない(動かすと、その位置の余裕
    ではなく、もっと広い隣の位置の余裕を測ってしまうため)。

    clearance_fn(pos, quat) は1姿勢の余裕(cm)。heading_deg は中心線の進行方向のヨー(度)。
    戻り値: (余裕, (pos, quat))。元より悪くはならない。
    """
    pos = np.asarray(pos, dtype=float)
    hr = np.radians(heading_deg)
    side = np.array([-np.sin(hr), np.cos(hr), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    with warnings.catch_warnings():
        # 垂直に立てた姿勢(ピッチ90度)ではヨーとロールが分けられない警告が出るが、
        # どちらに割り振っても同じ姿勢なので無視してよい
        warnings.simplefilter("ignore", UserWarning)
        yaw, neg_pitch, roll = Rotation.from_quat(quat).as_euler("ZYX", degrees=True)
    x = np.array([0.0, 0.0, yaw, -neg_pitch, roll])   # 横, 高さ, ヨー, ピッチ, ロール
    steps = np.array([step_cm, step_cm, step_deg, step_deg, step_deg])

    def pose_of(v):
        return pos + side * v[0] + up * v[1], _pose(v[2], v[3], v[4]).as_quat()

    best = clearance_fn(*pose_of(x))
    evals = 1
    while evals < max_evals and np.max(steps) >= min_step:
        moved = False
        for i in range(len(x)):
            for d in (1.0, -1.0):
                cand = x.copy()
                cand[i] += d * steps[i]
                c = clearance_fn(*pose_of(cand))
                evals += 1
                if c > best:
                    best, x, moved = c, cand, True
                    break
        if not moved:
            steps /= 2.0
    p_best, q_best = pose_of(x)
    return float(best), (p_best.tolist(), q_best.tolist())


def _lowest_positions(sweep, window_cm, max_n):
    """掃引結果のうち、一番狭い所から window_cm 以内で、余裕の小さい順に
    最大 max_n 個の弧長の集合。

    細かく掃引し直す位置を選ぶのに使う。余裕がほぼ一定の区間が長いと
    window_cm 以内の位置が何十個にもなるので、数を max_n で抑える。
    """
    cand = sorted((c, s) for s, c, _ in sweep if np.isfinite(c))
    if not cand:
        return set()
    worst = cand[0][0]
    return {s for c, s in cand[:max_n] if c <= worst + window_cm}


def refine_sweep(sweep, clearance_fn, heading_fn, max_n=40):
    """掃引結果の一番狭い所を refine_pose で探し直し、探し直した結果が
    一番狭い所でなくなったら、次に狭い所を探し直す。一番狭い所が探し直し
    済みの位置になったら止める(同じ形のリストを返す)。

    探し直すと余裕は増える一方なので、探し直していない位置が粗い値のまま
    一番狭い所に残ると、それが結果になってしまう。逆に余裕がほぼ一定の
    区間で近い値の位置を全部探し直すと、1条件に20分以上かかった。
    この順番なら、結果に効く位置だけを探し直せる。max_n は探し直す位置の上限。
    heading_fn(s) は弧長sでの中心線の進行方向のヨー(度)。
    """
    out = list(sweep)
    done = set()
    for _ in range(max_n):
        cand = [(c, k) for k, (_, c, _) in enumerate(out) if np.isfinite(c)]
        if not cand:
            break
        _, k = min(cand)
        if k in done:
            break
        s, c, pose = out[k]
        c2, pose2 = refine_pose(pose[0], pose[1], clearance_fn, heading_fn(s))
        if c2 > c:
            out[k] = (s, c2, pose2)
        done.add(k)
    return out


def fine_sweep(sweep_fn, clearance_fn, heading_fn, ds, ds_fine=3.0, window_cm=10.0,
               max_n=3):
    """粗い掃引のあと、一番狭い所の近くだけを細かく掃引し直して姿勢も探し直す。

    sweep_fn(s_values) は指定した弧長だけを掃引する関数(s_values=None なら
    いつもの刻みで全体)。粗い刻み(ds)の間に一番狭い所が隠れていることが
    あるので(L字で ds=15cm と 5cm で値が約10cm違った)、一番狭い所に近い
    max_n 個の位置の前後 ds の範囲を ds_fine 刻みで足し、そのうえで
    refine_sweep をかける。戻り値は弧長順の掃引結果(sweep_capacity と同じ形)。
    """
    coarse = sweep_fn(None)
    targets = _lowest_positions(coarse, window_cm, max_n)
    if not targets:
        return coarse
    have = {round(s, 6) for s, _, _ in coarse}
    s_lo, s_hi = coarse[0][0], coarse[-1][0]
    extra = set()
    for s in targets:
        for t in np.arange(s - ds + ds_fine, s + ds - 1e-9, ds_fine):
            if s_lo <= t <= s_hi and round(t, 6) not in have:
                extra.add(round(float(t), 6))
    merged = sorted(coarse + (sweep_fn(sorted(extra)) if extra else []),
                    key=lambda r: r[0])
    return refine_sweep(merged, clearance_fn, heading_fn)


def skeleton_heading(s):
    """弧長sでのL字の中心線の進行方向のヨー(度)。sweep_capacity の横方向と
    同じく、コーナー点の手前は90度(+y)、先は0度(+x)とする。"""
    return 90.0 if s <= S_CORNER else 0.0


def find_max_furniture_width(outer, obstacles, lo=None, hi=None, tol=1.0,
                             max_iter=6000, seeds=(0, 1)):
    """家具単体で通る最大の家具幅(cm)を二分探索する。

    phase1のfind_critical_widthと同じグローバル上書きパターンで、
    FURN_Wを一時的に変えながらRRTを走らせる。RRTは確率的なので、
    seedsのいずれかで経路が見つかればその幅は「通る」、どのseedでも
    見つからなければ「通らない」扱いにする(見つけ損ないは上限を
    小さめに見積もる側、つまり安全側に倒れる)。

    ボトルネック掃引(sweep_capacity)で代用しない理由: 掃引は位置ごとに
    「最良の姿勢」を取るため、踊り場での回転の途中で必ず通る中間姿勢の
    ピンチが見えず、幅の上限を過大評価する。

    戻り値: (max_w, trials)  trials = [(w, found), ...]
    """
    global FURN_W
    orig = FURN_W
    if lo is None:
        lo = FURN_W  # 呼び出し時点の幅で通ることが分かっている前提
    if hi is None:
        # 断面対角=階段幅となる解析上限(これ以上は垂直に立てても
        # 幅の帯の中で回せない)に、L字ポケットの分の余裕を足した値
        hi = float(np.sqrt(STAIR_WIDTH ** 2 - FURN_H ** 2)) + 8.0
    trials = []
    try:
        while hi - lo > tol:
            w = round((lo + hi) / 2.0, 1)
            FURN_W = w
            found = False
            for sd in seeds:
                path = p.rrt_connect(
                    START, GOAL, with_human=False, outer=outer, obstacles=obstacles,
                    max_iter=max_iter, seed=sd, w_rot=W_ROT,
                    sampler=make_sampler(with_human=False), validator=state_valid)
                if path is not None:
                    found = True
                    break
            trials.append((w, found))
            if found:
                lo = w
            else:
                hi = w
    finally:
        FURN_W = orig
    return lo, trials


def find_max_furniture_length(outer, obstacles, num_carriers=None, tol=2.0,
                              max_iter=4000, seeds=(0, 1), probe_los=(130.0, 100.0, 80.0)):
    """運搬者ありで通る最大の家具長さ(cm)を二分探索する。

    find_max_furniture_widthの長さ版。あちらは「家具単体」の上限幅
    だが、PRDの本命は「人が運ぶ前提での上限サイズ」なので、
    こちらは運搬者ありのオラクルで探索する。詰まりの主因は長さ
    (運搬者2人で実効長が約+100cmになる)なので、振る次元も長さにする。

    二分探索には「通る」下限が要るが、運搬者ありではどの長さが通るか
    事前に分からないため、probe_losを上から順に試して最初に通った
    長さをloにする。どれも通らなければ(None, trials)を返す
    (=この階段は人が運ぶ前提だと probe_los の最小値でも通らない)。
    hiは呼び出し時点のFURN_L(メイン実行でBLOCKED確認済みの長さ)。
    """
    global FURN_L
    orig = FURN_L
    hi = FURN_L
    trials = []

    def passes(length):
        global FURN_L
        FURN_L = length
        for sd in seeds:
            path = p.rrt_connect(
                START, GOAL, with_human=True, outer=outer, obstacles=obstacles,
                num_carriers=num_carriers, max_iter=max_iter, seed=sd, w_rot=W_ROT,
                sampler=make_sampler(with_human=True), validator=state_valid)
            if path is not None:
                return True
        return False

    try:
        lo = None
        for cand in probe_los:
            ok = passes(cand)
            trials.append((cand, ok))
            if ok:
                lo = cand
                break
        if lo is None:
            return None, trials
        while hi - lo > tol:
            m = round((lo + hi) / 2.0, 1)
            ok = passes(m)
            trials.append((m, ok))
            if ok:
                lo = m
            else:
                hi = m
    finally:
        FURN_L = orig
    return lo, trials


# ---- START / GOAL ----
_YAW90 = _pose(90.0, 0.0).as_quat()   # 長軸を+y(下廊下の進行方向)へ
_YAW0 = _pose(0.0, 0.0).as_quat()     # 長軸を+x(上廊下の進行方向)へ
START = (np.array([CENTER, 170.0, FURN_H / 2]), _YAW90)
GOAL = (np.array([FL2_X1 + 170.0, LAND_YC, TOP_Z + FURN_H / 2]), _YAW0)


def best_effort_path(tree):
    """RRTが失敗したとき、start側ツリーで「一番先まで進めた」ノードまでの
    経路を返す(進捗 = 中心線への射影の弧長)。phase1_gifのbest-effortと
    同じ役割で、演出ではなく実際に探索が到達できた最遠の状態。"""
    progress = [skeleton_s(n.pos[0], n.pos[1]) for n in tree]
    i = int(np.argmax(progress))
    path = []
    while i is not None:
        n = tree[i]
        path.append((n.pos, n.quat))
        i = n.parent
    path.reverse()
    return path


def horizontal_clearance_points(points, outer, obstacles):
    """床・天井方向を除いた「横方向の余裕」(cm)。点群の最悪値を返す。

    家具は床に接して滑るため、通常のクリアランス(全方向の符号付き
    距離)は経路上ほぼ常に0になり、「通る場合の余裕◯cm」の数字として
    意味をなさない。ここでは各点について「同じ高さのまま水平に
    どれだけ動かすと壁・段に当たるか」を測る:

      - 障害物は、その点のzが直方体のz範囲に(端を除いて)入っている
        場合だけ、xy平面での2D符号付き距離を数える。z範囲外の段は
        水平移動では当たらないので無視(家具が段の上面に載っている
        だけの接触も、z=上面ちょうどなので除外される)
      - 外枠(合併)は、zが範囲内の直方体のxy矩形の「内側の余裕」の最大値

    直方体は軸平行でも、鉛直軸(z)まわりに回転していてもよい(回り階段の
    扇形の段を回転した直方体で組むため)。点を各直方体のローカル座標に
    移してから同じ計算をする。z軸自体が傾いた直方体は「水平」の意味が
    崩れるので受け付けない。
    """
    eps = 1e-6
    pts = np.asarray(points, dtype=float).reshape(-1, 3)

    def local(c, r):
        if not np.allclose(r[:, 2], (0.0, 0.0, 1.0)):
            raise ValueError("horizontal_clearance_points: z軸まわり以外に回転した直方体は未対応")
        return (pts - c) @ r   # 各行 = R^T (p - c)

    inner = np.full(len(pts), -np.inf)
    for c, r, h in outer:
        q = local(c, r)
        m = np.minimum(h[0] - np.abs(q[:, 0]), h[1] - np.abs(q[:, 1]))
        inner = np.where(np.abs(q[:, 2]) <= h[2], np.maximum(inner, m), inner)
    v = inner
    for c, r, h in obstacles:
        q = local(c, r)
        qx = np.abs(q[:, 0]) - h[0]
        qy = np.abs(q[:, 1]) - h[1]
        d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0)
        v = np.where(np.abs(q[:, 2]) < h[2] - eps, np.minimum(v, d), v)
    return float(np.min(v))


def path_min_horizontal_clearance(path, outer, obstacles):
    """経路上で最も厳しい横方向の余裕(cm)と、そのインデックス(家具のみ)。"""
    vals = [horizontal_clearance_points(furniture_points(pos, q), outer, obstacles)
            for pos, q in path]
    i = int(np.argmin(vals))
    return vals[i], i


def path_min_clearance_lstair(path, with_human, outer, obstacles, num_carriers=None):
    """経路上で最も厳しい(最小の)クリアランス(cm)とそのインデックス。"""
    vals = []
    for pos, q in path:
        if with_human:
            vals.append(carriable_clearance_lstair(pos, q, outer, obstacles, num_carriers)[0])
        else:
            vals.append(furniture_clearance(pos, q, outer, obstacles))
    i = int(np.argmin(vals))
    return vals[i], i


def _where_on_skeleton(s):
    """弧長sの場所の英語ラベル(図の注記用)。踊り場ならlanding。"""
    s_land0 = FL1_Y1
    s_land1 = S_CORNER + (FL2_X0 - CENTER)
    return "landing" if s_land0 <= s <= s_land1 else f"s={s:.0f}cm"


def result_summary_lines(result):
    """結果JSON(dict)から、図に載せる結論の1行サマリー(英語)を組み立てる。

    PNG/GIF/3Dビューアが同じ数字を出すよう、出典をこのJSON一本に固定する
    ための共通関数。まだ計算していない項目(キーが無い)は黙って省く。
    図中の文字が英語なのはmatplotlib既定フォントの豆腐対策
    (phase2_lstair_viz.pyのdocstring参照)。
    """
    lines = []
    bn_box = result.get("bottleneck_furniture_only")
    if bn_box:
        lines.append(f"furniture alone: tightest at {_where_on_skeleton(bn_box['s'])}, "
                     f"{bn_box['capacity_cm']:.1f}cm margin at best pose")
    bn = result.get("bottleneck")
    if bn and bn["capacity_cm"] < 0:
        nc = result["meta"]["carrier"]["num_carriers"]
        tilt = result["meta"]["carrier"]["max_tilt_deg"]
        lines.append(f"with {nc} carriers: {-bn['capacity_cm']:.1f}cm short at "
                     f"{_where_on_skeleton(bn['s'])} (best pose within {tilt:.0f} deg tilt)")
    caps = []
    mw = result.get("max_width_furniture_only")
    if mw:
        caps.append(f"alone up to W={mw['max_w']:.0f}cm")
    ml = result.get("max_length_with_carriers")
    if ml and ml.get("max_l") is not None:
        caps.append(f"carried by {ml['num_carriers']} up to L={ml['max_l']:.0f}cm")
    if caps:
        lines.append("max furniture size for this staircase: " + " / ".join(caps))
    return lines


def _path_to_json(path):
    return [[*map(float, pos), *map(float, quat)] for pos, quat in path]


def result_meta(num_carriers, max_iter, seed):
    return {
        "stair": {"width": STAIR_WIDTH, "rise": RISE, "tread": TREAD,
                  "n_steps1": N_STEPS1, "n_steps2": N_STEPS2,
                  "landing": [STAIR_WIDTH, STAIR_WIDTH], "landing_z": LAND_Z,
                  "base_d": BASE_D, "top_d": TOP_D, "ceil_clear": CEIL_CLEAR},
        "furniture": {"L": FURN_L, "W": FURN_W, "H": FURN_H},
        "carrier": {"r": p.HUMAN_R, "height": p.HUMAN_HEIGHT, "arm": p.CARRY_ARM,
                    "leg_clear": LEG_CLEAR,
                    "reach": [REACH_MIN, REACH_MAX], "max_tilt_deg": MAX_TILT_DEG,
                    "num_carriers": num_carriers, **body_meta()},
        "planner": {"max_iter": max_iter, "seed": seed, "w_rot": W_ROT},
    }


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carriers", type=int, choices=(1, 2), default=p.NUM_CARRIERS)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-sweep", action="store_true",
                        help="ボトルネック掃引を省く(動作確認用)")
    parser.add_argument("--find-max-width", action="store_true",
                        help="家具単体で通る最大の家具幅を二分探索し、"
                             "既存の結果JSONに追記して終了する")
    parser.add_argument("--find-max-length", action="store_true",
                        help="運搬者ありで通る最大の家具長さを二分探索し、"
                             "既存の結果JSONに追記して終了する")
    parser.add_argument("--sweep-box", action="store_true",
                        help="家具単体のボトルネック掃引(通る場合の「最も狭い"
                             "場所で余裕◯cm」)を実行し、既存の結果JSONに追記して"
                             "終了する")
    parser.add_argument("--out", default=os.path.join("results", "lstair_result.json"))
    args = parser.parse_args()
    num_carriers = args.carriers

    outer, obstacles = build_lstairs()
    print(f"L字階段: 幅{STAIR_WIDTH:.0f}cm, 蹴上げ{RISE:.0f}cm x ({N_STEPS1}+{N_STEPS2})段, "
          f"踊り場{STAIR_WIDTH:.0f}x{STAIR_WIDTH:.0f}cm / "
          f"家具: {FURN_L:.0f}x{FURN_W:.0f}x{FURN_H:.0f}cm / 運搬者: {num_carriers}人")

    if args.find_max_width:
        import sys
        with open(args.out) as f:
            result = json.load(f)
        print("\n家具単体で通る最大幅を二分探索中(1回のRRTに数十秒かかる)...")
        t0 = time.time()
        max_w, trials = find_max_furniture_width(outer, obstacles)
        for w, ok in trials:
            print(f"  幅{w:5.1f}cm: {'PASS' if ok else 'BLOCKED'}")
        print(f"  -> この階段なら(長さ{FURN_L:.0f}・高さ{FURN_H:.0f}の家具単体は)"
              f"幅{max_w:.0f}cmまで入る [{time.time() - t0:.0f}s]")
        result["max_width_furniture_only"] = {
            "max_w": float(max_w),
            "trials": [[float(w), bool(ok)] for w, ok in trials],
            "note": "RRT2seedsで見つかればPASS。確率的探索なので下振れし得る",
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"updated {args.out}")
        sys.exit(0)

    if args.sweep_box:
        import sys
        with open(args.out) as f:
            result = json.load(f)
        # 通る場合の「余裕◯cm」は、RRTが見つけた生の経路から取ると
        # ほぼ常に0になる(経路は壁や床ギリギリを掠めて通るため。床に
        # 接して滑る分だけでも0)。ユーザーに見せる余裕は「その場所で
        # 一番良い姿勢を取ったときの余裕」なので、運搬者ありの
        # ボトルネックと同じ掃引を家具単体でも行う。
        print("\nボトルネック掃引(家具単体、中心線に沿って)...")
        t0 = time.time()
        sweep = sweep_capacity(outer, obstacles, with_human=False, fine=True)
        s_min, c_min, pose_min = min(sweep, key=lambda r: r[1])
        print(f"  最も狭い位置: 弧長 s={s_min:.0f}cm, 最良姿勢での余裕={c_min:.1f}cm "
              f"[{time.time() - t0:.0f}s]")
        result["bottleneck_furniture_only"] = {
            "s": float(s_min), "capacity_cm": float(c_min),
            "xy": list(map(float, skeleton_xy(s_min))),
            "pose": pose_min,
            "profile": [[float(s), float(c)] for s, c, _ in sweep],
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"updated {args.out}")
        sys.exit(0)

    if args.find_max_length:
        import sys
        with open(args.out) as f:
            result = json.load(f)
        print(f"\n運搬者{num_carriers}人で通る最大長さを二分探索中"
              "(1回のRRTに数十秒〜数分かかる)...")
        t0 = time.time()
        max_l, trials = find_max_furniture_length(
            outer, obstacles, num_carriers=num_carriers, max_iter=args.max_iter)
        for l, ok in trials:
            print(f"  長さ{l:5.1f}cm: {'PASS' if ok else 'BLOCKED'}")
        if max_l is None:
            print(f"  -> どの試行長さでも通らなかった [{time.time() - t0:.0f}s]")
        else:
            print(f"  -> この階段なら(幅{FURN_W:.0f}・高さ{FURN_H:.0f}の家具を"
                  f"{num_carriers}人で運ぶ前提で)長さ{max_l:.0f}cmまで入る "
                  f"[{time.time() - t0:.0f}s]")
        result["max_length_with_carriers"] = {
            "max_l": None if max_l is None else float(max_l),
            "num_carriers": num_carriers,
            "trials": [[float(l), bool(ok)] for l, ok in trials],
            "note": "RRT2seedsで見つかればPASS。確率的探索なので下振れし得る",
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"updated {args.out}")
        sys.exit(0)

    for label, (pos, quat) in (("START", START), ("GOAL", GOAL)):
        bc = furniture_clearance(pos, quat, outer, obstacles)
        cc, _ = carriable_clearance_lstair(pos, quat, outer, obstacles, num_carriers)
        print(f"  {label}: 家具単体={bc:.1f}cm / 運搬者あり={cc:.1f}cm")

    result = {"meta": result_meta(num_carriers, args.max_iter, args.seed)}

    print("\n家具単体の経路を探索中...")
    t0 = time.time()
    path_box, tree_box = p.rrt_connect(
        START, GOAL, with_human=False, outer=outer, obstacles=obstacles,
        max_iter=args.max_iter, seed=args.seed, w_rot=W_ROT,
        sampler=make_sampler(with_human=False), validator=state_valid, return_trees=True)
    dt_box = time.time() - t0
    if path_box is not None:
        mc, _ = path_min_clearance_lstair(path_box, False, outer, obstacles)
        print(f"  PASS (min clearance {mc:.1f}cm) [{dt_box:.0f}s]")
        result["box_only"] = {"found": True, "path": _path_to_json(path_box),
                              "min_clearance": mc, "time_s": dt_box}
    else:
        print(f"  BLOCKED (この試行回数では見つからず) [{dt_box:.0f}s]")
        result["box_only"] = {"found": False, "time_s": dt_box,
                              "best_effort_path": _path_to_json(best_effort_path(tree_box))}

    print(f"\n運搬者あり({num_carriers}人)の経路を探索中...")
    t0 = time.time()
    path_h, tree_h = p.rrt_connect(
        START, GOAL, with_human=True, outer=outer, obstacles=obstacles,
        num_carriers=num_carriers, max_iter=args.max_iter, seed=args.seed, w_rot=W_ROT,
        sampler=make_sampler(with_human=True), validator=state_valid, return_trees=True)
    dt_h = time.time() - t0
    if path_h is not None:
        mc, _ = path_min_clearance_lstair(path_h, True, outer, obstacles, num_carriers)
        print(f"  PASS (min clearance {mc:.1f}cm) [{dt_h:.0f}s]")
        result["with_carriers"] = {"found": True, "path": _path_to_json(path_h),
                                   "min_clearance": mc, "time_s": dt_h}
    else:
        be = best_effort_path(tree_h)
        s_reach = skeleton_s(be[-1][0][0], be[-1][0][1])
        print(f"  BLOCKED (この試行回数では見つからず; 弧長{s_reach:.0f}cmまで到達) [{dt_h:.0f}s]")
        result["with_carriers"] = {"found": False, "time_s": dt_h,
                                   "reached_s": float(s_reach),
                                   "best_effort_path": _path_to_json(be)}

    if not args.skip_sweep:
        print("\nボトルネック掃引(運搬者あり、中心線に沿って)...")
        t0 = time.time()
        sweep = sweep_capacity(outer, obstacles, num_carriers=num_carriers, fine=True)
        finite = [(s, c, pose) for s, c, pose in sweep if np.isfinite(c)]
        s_min, c_min, pose_min = min(finite, key=lambda r: r[1])
        print(f"  最悪の位置: 弧長 s={s_min:.0f}cm (踊り場={S_CORNER - STAIR_WIDTH / 2:.0f}"
              f"..{S_CORNER + STAIR_WIDTH / 2:.0f}cm), 余裕={c_min:.1f}cm [{time.time() - t0:.0f}s]")
        if c_min < 0:
            print(f"  -> この位置では、傾き{MAX_TILT_DEG:.0f}度以内のどの姿勢でも"
                  f"あと{-c_min:.1f}cm足りない(姿勢グリッドの範囲で)")
        bd = clearance_breakdown(np.array(pose_min[0]), np.array(pose_min[1]),
                                 outer, obstacles, num_carriers)
        print("  " + format_breakdown(bd))
        result["bottleneck"] = {
            "s": float(s_min), "capacity_cm": float(c_min),
            "xy": list(map(float, skeleton_xy(s_min))),
            "pose": pose_min,
            "breakdown": bd,
            "profile": [[float(s), float(c)] for s, c, _ in sweep],
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nsaved result to {args.out}")
    print("注意: RRTは確率的探索なので、BLOCKEDは「この試行回数では見つからなかった」の意味。"
          "ボトルネック掃引の不足量が負であることが「通らない」ことの裏付けになる。")
