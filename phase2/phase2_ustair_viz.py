"""
折り返し(U字)階段の説明用の図: results/ustair_result.json から1枚のPNGを作る。

    python phase2_ustair_viz.py                        # phase2_ustair.png
    python phase2_ustair_viz.py --out ../demo/ustair_explain.png

初めて見る人(発表・展示)に向けた図なので、文字は日本語にする。
L字の図(phase2_lstair_viz.py)が英語なのは豆腐対策だったが、Windows標準の
日本語フォント(BIZ UDPGothic / Meiryo / Yu Gothic)があればそれを使う。
見つからない環境では警告を出す(図は描けるが日本語が豆腐になる)。

構成(左から):
  ① 上から見た図・家具だけ   -- 通った経路の残像
  ② 上から見た図・2人で運ぶ -- 探索が進めた所までの残像と、踊り場で
     「一番うまい向きにしても足りない」姿勢を赤で
  ③ 余裕のグラフ -- 横軸=玄関からの道のり、縦軸=壁・段までの余裕(cm)。
     0より下は「どう向けてもぶつかる」。L字(同じ家具・同じ人数)を
     点線で重ね、折り返しでは踊り場全体がマイナスのままになることを見せる

探索は再実行しない(JSONの経路と掃引結果を描くだけ)。
"""

import argparse
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from scipy.spatial import ConvexHull

import phase2_lstair as L
import phase2_ustair as U

JP_FONTS = ("Meiryo", "BIZ UDPGothic", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP")

FREE_C = "#f1f1f1"
STEP_C = "#d4d4d4"
STEP_E = "#a8a8a8"
WALL_C = "#8a8a8a"
LAND_C = "#fff3cc"
OK_C = "#2c6fbb"
NG_C = "#c0392b"
START_C = "#2e9e5b"
PERSON_C = "#e07b39"


def use_japanese_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    found = [f for f in JP_FONTS if f in have]
    if not found:
        warnings.warn("日本語フォントが見つからないため、図の日本語が豆腐になる可能性があります")
    plt.rcParams["font.family"] = found + ["DejaVu Sans"]


def parse_path(raw):
    return [(np.array(s[:3]), np.array(s[3:])) for s in raw]


def furniture_top(pos, quat):
    pts = L.furniture_points(pos, quat)[:, :2]
    return pts[ConvexHull(pts).vertices]


# ---- 上から見た図 ----

def draw_env_top(ax):
    """床(廊下・踊り場)、段、上下フライトの間の壁、入口とゴールの目印。"""
    W = U.STAIR_WIDTH
    ax.add_patch(patches.Rectangle((0, 0), W, U.LAND_Y1, facecolor=FREE_C, zorder=0))
    ax.add_patch(patches.Rectangle((U.X_R0, U.TOP_Y0), W, U.LAND_Y1 - U.TOP_Y0,
                                   facecolor=FREE_C, zorder=0))
    ax.add_patch(patches.Rectangle((0, U.FL1_Y1), U.X_R1, W, facecolor=LAND_C, zorder=0.1))
    ax.add_patch(patches.Rectangle((W, U.TOP_Y0), U.WALL, U.FL1_Y1 - U.TOP_Y0,
                                   facecolor=WALL_C, zorder=0.4))
    for i in range(U.N_STEPS1):
        ax.add_patch(patches.Rectangle((0, U.FL1_Y0 + U.TREAD * i), W, U.TREAD,
                                       facecolor=STEP_C, edgecolor=STEP_E, zorder=0.3))
    for i in range(U.N_STEPS2):
        ax.add_patch(patches.Rectangle((U.X_R0, U.FL1_Y1 - U.TREAD * (i + 1)), W, U.TREAD,
                                       facecolor=STEP_C, edgecolor=STEP_E, zorder=0.3))
    # 踊り場の奥の壁の向こう(実体)。詰まる姿勢のはみ出しが見えるように帯で塗る
    ax.add_patch(patches.Rectangle((0, U.LAND_Y1), U.X_R1, 45, facecolor="#e2e2e2",
                                   hatch="///", edgecolor="#b5b5b5", linewidth=0, zorder=0))
    ax.text(U.X_R1 / 2, U.LAND_Y1 + 22, "壁の向こう", ha="center", va="center",
            fontsize=9, color="#777777", zorder=0.6)
    # 外周の壁の線
    ax.plot([0, 0, U.X_R1, U.X_R1], [0, U.LAND_Y1, U.LAND_Y1, U.TOP_Y0],
            color=WALL_C, linewidth=2.5, zorder=0.5)
    ax.text(U.X_R1 / 2, U.LAND_YC, "踊り場", ha="center", va="center",
            fontsize=11, color="#a08020", fontweight="bold", zorder=0.6)
    ax.annotate("", (U.CX1, U.FL1_Y1 - 10), (U.CX1, U.FL1_Y0 + 10),
                arrowprops=dict(arrowstyle="->", color="#777777", lw=1.5), zorder=0.6)
    ax.annotate("", (U.CX2, U.FL2_Y1 + 10), (U.CX2, U.FL1_Y1 - 10),
                arrowprops=dict(arrowstyle="->", color="#777777", lw=1.5), zorder=0.6)
    ax.text(U.CX1, -28, "入口(1階)", ha="center", va="top", fontsize=10)
    ax.text(U.CX2, -28, "ゴール(2階)", ha="center", va="top", fontsize=10)
    ax.set_xlim(-20, U.X_R1 + 20)
    ax.set_ylim(-75, U.LAND_Y1 + 50)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_furniture(ax, pos, quat, color, alpha, zorder=2, lw=0.6):
    ax.add_patch(patches.Polygon(furniture_top(pos, quat), closed=True, facecolor=color,
                                 edgecolor="k", alpha=alpha, linewidth=lw, zorder=zorder))


def draw_people(ax, pos, quat, num_carriers, color, alpha, zorder=2.5):
    outer, obstacles = U.build_ustairs()
    for c in L.best_human_positions_lstair(pos, quat, num_carriers, outer, obstacles,
                                           floor_fn=U.floor_z):
        ax.add_patch(patches.Circle((c[0], c[1]), L.p.HUMAN_R, facecolor=color,
                                    edgecolor="k", alpha=alpha, linewidth=0.6, zorder=zorder))


def draw_trail(ax, path, num_carriers=0, n=14):
    """経路の残像。最初の姿勢は緑、最後は濃い青。"""
    idx = np.unique(np.round(np.linspace(0, len(path) - 1, n)).astype(int))
    for i in idx:
        draw_furniture(ax, *path[i], OK_C, 0.12)
        if num_carriers:
            draw_people(ax, *path[i], num_carriers, PERSON_C, 0.10)
    draw_furniture(ax, *path[0], START_C, 0.85, zorder=3)
    if num_carriers:
        draw_people(ax, *path[0], num_carriers, PERSON_C, 0.8, zorder=3.1)


# ---- 余裕のグラフ ----

def draw_margin_chart(ax, u, l):
    s_land0 = U.FL1_Y1
    s_land1 = U.S2 + (U.LAND_YC - U.FL1_Y1)
    ax.axvspan(s_land0, s_land1, color=LAND_C, zorder=0)
    ax.text((s_land0 + s_land1) / 2, 31, "折り返しの踊り場", ha="center",
            fontsize=10, color="#a08020", fontweight="bold")
    ax.axhspan(-40, 0, color="#fbe3e0", zorder=0.1)
    ax.axhline(0, color="k", linewidth=1)
    ax.text(260, -37, "0より下 = どう向けても壁や段にぶつかる", fontsize=10,
            color=NG_C, va="bottom")

    def xs_ys(profile):
        a = np.array(profile)
        return a[:, 0], np.clip(a[:, 1], -40, 40)

    x, y = xs_ys(u["bottleneck_furniture_only"]["profile"])
    ax.plot(x, y, color=OK_C, linewidth=2.2, label="折り返し・家具だけ")
    x, y = xs_ys(l["bottleneck"]["profile"])
    ax.plot(x, y, color="#777777", linewidth=1.8, linestyle="--", label="L字・2人で運ぶ(比較)")
    x, y = xs_ys(u["bottleneck"]["profile"])
    ax.plot(x, y, color=NG_C, linewidth=2.6, label="折り返し・2人で運ぶ")

    bn = u["bottleneck"]
    ax.annotate(f"最大 {-bn['capacity_cm']:.0f}cm 足りない", (bn["s"], bn["capacity_cm"]),
                xytext=(bn["s"] - 250, -24), fontsize=11, color=NG_C, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NG_C))
    # L字は踊り場の出口(弧長約595cm)で余裕がプラスに戻る
    lp = np.array(l["bottleneck"]["profile"])
    rec = lp[(lp[:, 0] > s_land0) & (lp[:, 1] > 0)]
    if len(rec):
        ax.annotate("L字はここで\nプラスに戻る", (rec[0, 0], rec[0, 1]),
                    xytext=(rec[0, 0] + 25, 22), fontsize=9.5, color="#555555",
                    arrowprops=dict(arrowstyle="->", color="#777777"))
    ax.annotate("折り返しは踊り場の間\n一度も0を超えない", (640, -29),
                xytext=(715, -22), fontsize=9.5, color=NG_C,
                arrowprops=dict(arrowstyle="->", color=NG_C))

    ax.set_xlim(250, 950)
    ax.set_ylim(-40, 40)
    ax.set_xlabel("入口からの道のり (cm)")
    ax.set_ylabel("壁・段までの余裕 (cm)")
    ax.set_title("③ 進むにつれて、余裕はどう変わるか", loc="left", fontsize=13)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.3)


def render_png(u, l, out_path):
    use_japanese_font()
    fu, st = u["meta"]["furniture"], u["meta"]["stair"]
    nc = u["meta"]["carrier"]["num_carriers"]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.3], wspace=0.08,
                          left=0.02, right=0.98, top=0.83, bottom=0.08)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ① 家具だけ
    draw_env_top(ax1)
    box = u["box_only"]
    path = parse_path(box["path"] if box["found"] else box["best_effort_path"])
    draw_trail(ax1, path)
    draw_furniture(ax1, *path[-1], OK_C, 0.85, zorder=3)
    m = u["bottleneck_furniture_only"]["capacity_cm"]
    ax1.set_title(f"① 家具だけ\n→ 通る(一番狭い所で余裕{m:.0f}cm)" if box["found"]
                  else "① 家具だけ\n→ 通らない", fontsize=13, color=OK_C, loc="left")

    # ② 2人で運ぶ
    draw_env_top(ax2)
    car = u["with_carriers"]
    path = parse_path(car["path"] if car["found"] else car["best_effort_path"])
    draw_trail(ax2, path, num_carriers=nc)
    bn = u["bottleneck"]
    if not car["found"]:
        pos, q = np.array(bn["pose"][0]), np.array(bn["pose"][1])
        draw_furniture(ax2, pos, q, NG_C, 0.85, zorder=4, lw=1.0)
        draw_people(ax2, pos, q, nc, NG_C, 0.75, zorder=4.1)
        # 最もはみ出している家具の点(奥の壁側)を矢印で指す
        pts = L.furniture_points(pos, q)
        tip = pts[np.argmax(pts[:, 1])]
        ax2.annotate(f"一番うまく向けても\n奥の壁に{-bn['capacity_cm']:.0f}cm\nはみ出す",
                     (tip[0], min(tip[1], U.LAND_Y1 + 40)),
                     xytext=(U.CX2 + 5, U.FL2_Y1 - 60), fontsize=10.5,
                     color=NG_C, fontweight="bold", ha="center",
                     bbox=dict(boxstyle="round", facecolor="white", edgecolor=NG_C, alpha=0.95),
                     arrowprops=dict(arrowstyle="->", color=NG_C, lw=1.5), zorder=5)
    ax2.set_title(f"② {nc}人で運ぶ\n→ " + ("通る" if car["found"] else
                  f"通らない(あと{-bn['capacity_cm']:.0f}cm)"),
                  fontsize=13, color=OK_C if car["found"] else NG_C, loc="left")

    # ③ 余裕のグラフ
    draw_margin_chart(ax3, u, l)

    fig.suptitle("折り返し階段:  家具だけなら通るのに、人が運ぶと踊り場で回せない",
                 fontsize=18, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.905,
             f"階段 幅{st['width']:.0f}cm・{st['n_steps1']}+{st['n_steps2']}段 / "
             f"家具(タンス) {fu['L']:.0f}×{fu['W']:.0f}×{fu['H']:.0f}cm / "
             f"運ぶ人は直径{2 * L.p.HUMAN_R:.0f}cmの円柱で近似、家具の傾きは"
             f"{u['meta']['carrier']['max_tilt_deg']:.0f}度まで  "
             "(緑=出発時の姿勢、青=通った跡、橙=運ぶ人、赤=詰まる姿勢)",
             fontsize=10.5, color="#444444")
    fig.savefig(out_path, dpi=110)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=os.path.join("results", "ustair_result.json"))
    parser.add_argument("--compare", default=os.path.join("results", "lstair_result.json"),
                        help="比較に重ねるL字の結果JSON")
    parser.add_argument("--out", default="phase2_ustair.png")
    args = parser.parse_args()
    with open(args.json) as f:
        u = json.load(f)
    with open(args.compare) as f:
        l = json.load(f)
    # 描画に使う寸法を結果JSONに合わせる
    st, fu = u["meta"]["stair"], u["meta"]["furniture"]
    L.FURN_L, L.FURN_W, L.FURN_H = fu["L"], fu["W"], fu["H"]
    U.configure(width=st["width"], rise=st["rise"], tread=st["tread"],
                n_steps1=st["n_steps1"], n_steps2=st["n_steps2"], wall=st["wall"],
                base_d=st["base_d"], top_d=st["top_d"], ceil_clear=st["ceil_clear"])
    render_png(u, l, args.out)
