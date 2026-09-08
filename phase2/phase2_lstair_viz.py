"""
L字階段の可視化: phase2_lstair.py が保存した results/*.json から、
phase1_demo.py の run_case と同じ型の比較図を再生成する。

    python phase2_lstair_viz.py                       # results/lstair_result.json から
    python phase2_lstair_viz.py --json <path> --out <png>

1枚のPNGに 左=家具単体(PASS) / 右=運搬者あり(BLOCKED) を並べる。
3Dなので各ケースを 上段=上面図(x-y) / 下段=展開側面図(中心線に沿った
弧長s - z) の2段で描く。詰まる場合は最も厳しい位置の家具と運搬者を
赤で強調し、不足量(あと何cm)を注記する。

図中の文字は英語(matplotlibの既定フォントに日本語グリフがなく
豆腐になる環境が多いため。図の内容の説明はREADME側でする)。
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

import phase2_lstair as L


# ---- JSONの読み込み ----

def load_result(path):
    with open(path) as f:
        result = json.load(f)
    # 図はモジュール定数から環境を再構築して描くので、JSONが別の寸法で
    # 作られていたら図が嘘になる。ここで突き合わせて検出する。
    meta = result["meta"]
    assert meta["furniture"] == {"L": L.FURN_L, "W": L.FURN_W, "H": L.FURN_H}, \
        "JSONの家具寸法がphase2_lstair.pyの定数と一致しない"
    assert meta["stair"]["width"] == L.STAIR_WIDTH and meta["stair"]["rise"] == L.RISE, \
        "JSONの階段寸法がphase2_lstair.pyの定数と一致しない"
    return result


def parse_path(raw):
    """[[x,y,z,qx,qy,qz,qw], ...] -> [(pos, quat), ...]"""
    return [(np.array(s[:3]), np.array(s[3:])) for s in raw]


# ---- 2D投影ヘルパー ----

def _hull(points_2d):
    """凸包の頂点列(phase2_viz._order_ccwと同じ理由: 表面サンプル点を
    そのまま結ぶとジグザグするため)。"""
    pts = np.asarray(points_2d)
    return pts[ConvexHull(pts).vertices]


def furniture_poly_top(pos, quat):
    return _hull(L.furniture_points(pos, quat)[:, :2])


def furniture_poly_side(pos, quat):
    """展開側面図: 各表面点を(中心線への射影の弧長s, z)へ落とす。
    コーナー付近では投影がひずむが、定性的な読み取りには足りる。"""
    pts = L.furniture_points(pos, quat)
    sz = np.array([[L.skeleton_s(x, y), z] for x, y, z in pts])
    return _hull(sz)


# ---- 環境の描画 ----

FREE_C = "#ececec"
STEP_C = "#c9c9c9"
STEP_E = "#a8a8a8"


def draw_env_top(ax, title):
    """上面図(x-y): 廊下2本の帯と、各ステップの踏み面。"""
    ax.add_patch(patches.Rectangle((0, 0), L.STAIR_WIDTH, L.LAND_Y1,
                                   facecolor=FREE_C, edgecolor="none", zorder=0))
    ax.add_patch(patches.Rectangle((0, L.FL1_Y1), L.TOP_X1, L.STAIR_WIDTH,
                                   facecolor=FREE_C, edgecolor="none", zorder=0))
    for i in range(L.N_STEPS1):
        y0 = L.FL1_Y0 + L.TREAD * i
        ax.add_patch(patches.Rectangle((0, y0), L.STAIR_WIDTH, L.TREAD,
                                       facecolor=STEP_C, edgecolor=STEP_E, zorder=0.5))
    for i in range(L.N_STEPS2):
        x0 = L.FL2_X0 + L.TREAD * i
        ax.add_patch(patches.Rectangle((x0, L.FL1_Y1), L.TREAD, L.STAIR_WIDTH,
                                       facecolor=STEP_C, edgecolor=STEP_E, zorder=0.5))
    ax.annotate("landing\n(odoriba)", (L.STAIR_WIDTH / 2, L.LAND_YC),
                ha="center", va="center", fontsize=8, color="#666666")
    ax.annotate("up", (L.STAIR_WIDTH / 2, L.FL1_Y0 + 40), ha="center",
                fontsize=8, color="#666666", rotation=90)
    ax.annotate("up", (L.FL2_X0 + 40, L.LAND_YC), ha="center",
                fontsize=8, color="#666666")
    ax.set_xlim(-15, L.TOP_X1 + 15)
    ax.set_ylim(-15, L.LAND_Y1 + 15)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")


def draw_env_side(ax, title):
    """展開側面図(弧長s - z): 床のプロファイルと天井。"""
    # 床下(実体)を塗る
    ax.add_patch(patches.Rectangle((0, -12), L.S_TOTAL, 12,
                                   facecolor=STEP_C, edgecolor="none", zorder=0.5))
    for i in range(L.N_STEPS1):
        s0 = L.FL1_Y0 + L.TREAD * i
        ax.add_patch(patches.Rectangle((s0, 0), L.TREAD, L.RISE * (i + 1),
                                       facecolor=STEP_C, edgecolor=STEP_E, zorder=0.5))
    s_land0 = L.FL1_Y1
    s_land1 = L.S_CORNER + (L.FL2_X0 - L.CENTER)
    ax.add_patch(patches.Rectangle((s_land0, 0), s_land1 - s_land0, L.LAND_Z,
                                   facecolor=STEP_C, edgecolor=STEP_E, zorder=0.5))
    for i in range(L.N_STEPS2):
        s0 = s_land1 + L.TREAD * i
        ax.add_patch(patches.Rectangle((s0, 0), L.TREAD, L.LAND_Z + L.RISE * (i + 1),
                                       facecolor=STEP_C, edgecolor=STEP_E, zorder=0.5))
    s_top = s_land1 + L.N_STEPS2 * L.TREAD
    ax.add_patch(patches.Rectangle((s_top, 0), L.S_TOTAL - s_top, L.TOP_Z,
                                   facecolor=STEP_C, edgecolor=STEP_E, zorder=0.5))
    # 天井(下棟322cm / 踊り場から先424cm)と、その上の実体
    ceil_a = L.LAND_Z + L.CEIL_CLEAR
    ceil_b = L.TOP_Z + L.CEIL_CLEAR
    ax.add_patch(patches.Rectangle((0, ceil_a), L.FL1_Y1, ceil_b - ceil_a + 12,
                                   facecolor=STEP_C, edgecolor=STEP_E, zorder=0.5))
    ax.add_patch(patches.Rectangle((0, ceil_b), L.S_TOTAL, 12,
                                   facecolor=STEP_C, edgecolor="none", zorder=0.5))
    ax.axvspan(s_land0, s_land1, facecolor="#fff3cc", zorder=0.2)
    ax.annotate("landing", ((s_land0 + s_land1) / 2, ceil_b - 20),
                ha="center", fontsize=8, color="#a08020")
    ax.set_xlim(-15, L.S_TOTAL + 15)
    ax.set_ylim(-15, ceil_b + 25)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("s = arc length along centerline (cm)")
    ax.set_ylabel("z (cm)")


# ---- 家具・運搬者の描画 ----

def draw_furniture(ax_top, ax_side, pos, quat, color="tab:blue", alpha=0.6, zorder=2):
    for ax, poly in ((ax_top, furniture_poly_top(pos, quat)),
                     (ax_side, furniture_poly_side(pos, quat))):
        ax.add_patch(patches.Polygon(poly, closed=True, facecolor=color,
                                     edgecolor="k", alpha=alpha, linewidth=0.8,
                                     zorder=zorder))


def draw_carriers(ax_top, ax_side, pos, quat, outer, obstacles, num_carriers,
                  color="tab:orange", alpha=0.5, zorder=1.5):
    """運搬者: 上面図では円、側面図では胴体(膝上)の矩形+脚の線。"""
    for c in L.best_human_positions_lstair(pos, quat, num_carriers, outer, obstacles):
        x, y = c[0], c[1]
        ax_top.add_patch(patches.Circle((x, y), L.p.HUMAN_R, facecolor=color,
                                        edgecolor="k", alpha=alpha, zorder=zorder))
        s = L.skeleton_s(x, y)
        fz = L.floor_z(x, y)
        if fz is None:
            fz = c[2] - L._CARRIER_HALF_H - L.LEG_CLEAR
        # 胴体(オラクルが占有として扱う部分)
        ax_side.add_patch(patches.Rectangle(
            (s - L.p.HUMAN_R, fz + L.LEG_CLEAR), 2 * L.p.HUMAN_R,
            L.p.HUMAN_HEIGHT - L.LEG_CLEAR,
            facecolor=color, edgecolor="k", alpha=alpha, zorder=zorder))
        # 脚(オラクルでは無視する部分)は線だけ
        for dx in (-L.p.HUMAN_R / 2, L.p.HUMAN_R / 2):
            ax_side.plot([s + dx, s + dx], [fz, fz + L.LEG_CLEAR],
                         color="k", linewidth=1.0, alpha=alpha * 0.8, zorder=zorder)


def draw_case(ax_top, ax_side, path, with_human, outer, obstacles, num_carriers,
              ghost_n=12, worst=None, worst_label=None):
    """1ケース分(経路の残像 + START緑 + GOAL/最終青 + 詰まり赤)を描く。

    worst = (pos, quat) を渡すと赤で強調し、worst_labelを注記する。
    """
    step = max(1, len(path) // ghost_n)
    for pos, q in path[::step]:
        draw_furniture(ax_top, ax_side, pos, q, color="tab:blue", alpha=0.13)
        if with_human:
            draw_carriers(ax_top, ax_side, pos, q, outer, obstacles, num_carriers,
                          alpha=0.10)
    draw_furniture(ax_top, ax_side, *path[0], color="tab:green", alpha=0.9)
    draw_furniture(ax_top, ax_side, *path[-1], color="tab:blue", alpha=0.9)
    if with_human:
        draw_carriers(ax_top, ax_side, *path[0], outer, obstacles, num_carriers,
                      color="tab:green", alpha=0.45)
        draw_carriers(ax_top, ax_side, *path[-1], outer, obstacles, num_carriers,
                      color="tab:blue", alpha=0.45)
    if worst is not None:
        pos, q = worst
        draw_furniture(ax_top, ax_side, pos, q, color="tab:red", alpha=0.9, zorder=3)
        if with_human:
            draw_carriers(ax_top, ax_side, pos, q, outer, obstacles, num_carriers,
                          color="tab:red", alpha=0.7, zorder=2.8)
        if worst_label:
            # 注記はタイトルや経路と重ならない、図の空いている側に置く
            for ax, xy, off in ((ax_top, (pos[0], pos[1]), (70, -190)),
                                (ax_side, (L.skeleton_s(pos[0], pos[1]), pos[2]),
                                 (150, 140))):
                ax.annotate(worst_label, xy,
                            xytext=(xy[0] + off[0], xy[1] + off[1]),
                            fontsize=9, color="tab:red", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="tab:red"),
                            zorder=4)


def render_png(result, out_path):
    outer, obstacles = L.build_lstairs()
    meta = result["meta"]
    num_carriers = meta["carrier"]["num_carriers"]

    box = result["box_only"]
    car = result["with_carriers"]
    box_found = box["found"]
    car_found = car["found"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5),
                             gridspec_kw={"height_ratios": [1.15, 1]})
    (ax_t_box, ax_t_car), (ax_s_box, ax_s_car) = axes

    # 左: 家具単体。PASSの場合、経路上の最小クリアランスは床への接地
    # (ぎりぎり0cm)で決まることが多く、その状態を赤で塗るとSTARTが
    # 隠れて紛らわしいだけなので、数値はタイトルに載せるにとどめる。
    path_box = parse_path(box["path"] if box_found else box["best_effort_path"])
    box_title = "Furniture only: "
    if box_found:
        mc, _ = L.path_min_clearance_lstair(path_box, False, outer, obstacles)
        # 最小クリアランスは床への接地(=0cm)で決まりがちで、そのまま
        # 「余裕0.0cm」と書くと壁ギリギリと誤読される。接地由来である
        # ことをタイトルで明示する(床接触を除いた横方向余裕への置き換えは
        # 別途)。
        if mc + 0.0 < 0.5:
            box_title += "PASS (min clearance 0cm = floor contact while sliding)"
        else:
            box_title += f"PASS (min clearance {mc + 0.0:.1f}cm)"  # +0.0で-0.0表記を防ぐ
    else:
        box_title += "BLOCKED (not found)"
    path_car = parse_path(car["path"] if car_found else car["best_effort_path"])
    car_title = f"Furniture + {num_carriers} carriers: "
    if car_found:
        mc_car, _ = L.path_min_clearance_lstair(path_car, True, outer, obstacles,
                                                num_carriers)
        car_title += f"PASS (min clearance {mc_car + 0.0:.1f}cm)"
    else:
        car_title += "BLOCKED (no path found)"

    draw_env_top(ax_t_box, box_title + " -- top view")
    draw_env_top(ax_t_car, car_title + " -- top view")
    draw_env_side(ax_s_box, "unfolded side view (along centerline)")
    draw_env_side(ax_s_car, "unfolded side view (along centerline)")

    draw_case(ax_t_box, ax_s_box, path_box, False, outer, obstacles, num_carriers)

    # 右: 運搬者あり。BLOCKEDならボトルネック(掃引の最悪位置での
    # 一番マシな姿勢)を赤で強調し、不足量を注記する。
    worst_car = None
    label_car = None
    if not car_found and "bottleneck" in result:
        bn = result["bottleneck"]
        pos, quat = np.array(bn["pose"][0]), np.array(bn["pose"][1])
        worst_car = (pos, quat)
        deficit = -bn["capacity_cm"]
        label_car = (f"bottleneck: {deficit:.1f}cm short\n"
                     f"(best pose within {meta['carrier']['max_tilt_deg']:.0f}° tilt)")
    draw_case(ax_t_car, ax_s_car, path_car, True, outer, obstacles, num_carriers,
              worst=worst_car, worst_label=label_car)

    # 幅の二分探索(--find-max-width)の結果があれば、左パネルに
    # 「この階段なら幅◯cmまで入る」を注記する
    if "max_width_furniture_only" in result:
        mw = result["max_width_furniture_only"]["max_w"]
        ax_t_box.text(0.97, 0.03,
                      f"this staircase fits furniture\n"
                      f"up to W = {mw:.0f}cm\n"
                      f"(furniture alone, same L x H,\n"
                      f" bisection over RRT runs)",
                      transform=ax_t_box.transAxes, ha="right", va="bottom",
                      fontsize=9, color="#205020",
                      bbox=dict(boxstyle="round", facecolor="#eaf5ea",
                                edgecolor="#88aa88"))

    # 長さの二分探索(--find-max-length)の結果があれば、右パネルに
    # 「◯人で運ぶなら長さ◯cmまで」を注記する(運搬者ありの上限サイズ、
    # PRDの本命の出力)
    if "max_length_with_carriers" in result:
        ml = result["max_length_with_carriers"]
        if ml["max_l"] is not None:
            ax_t_car.text(0.97, 0.03,
                          f"carried by {ml['num_carriers']} people, this staircase\n"
                          f"fits furniture up to L = {ml['max_l']:.0f}cm\n"
                          f"(same W x H; furniture alone passes\n"
                          f" even at L = {meta['furniture']['L']:.0f}cm)",
                          transform=ax_t_car.transAxes, ha="right", va="bottom",
                          fontsize=9, color="#5a2020",
                          bbox=dict(boxstyle="round", facecolor="#f7ecec",
                                    edgecolor="#bb8888"))

    st = meta["stair"]
    fu = meta["furniture"]
    verdict = f"furniture alone {'PASS' if box_found else 'BLOCKED'} / " \
              f"with carriers {'PASS' if car_found else 'BLOCKED'}"
    fig.suptitle(
        "Odoriba Phase 2: L-shaped staircase "
        f"(width {st['width']:.0f}cm, rise {st['rise']:.0f} x tread {st['tread']:.0f}cm, "
        f"{st['n_steps1']}+{st['n_steps2']} steps, landing {st['landing'][0]:.0f}x{st['landing'][1]:.0f}cm) / "
        f"furniture {fu['L']:.0f}x{fu['W']:.0f}x{fu['H']:.0f}cm\n"
        f"result: {verdict}",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=os.path.join("results", "lstair_result.json"))
    parser.add_argument("--out", default="phase2_lstair.png")
    args = parser.parse_args()
    render_png(load_result(args.json), args.out)
