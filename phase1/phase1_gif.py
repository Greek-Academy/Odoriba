"""
Phase 1 GIF: 箱単体(ゴールに到達)vs 箱+運搬者(best-effort、
ゴールには到達しない)。

前後2カプセル+横位置探索の可搬性オラクルでは、箱単体の経路が
そのまま「運搬者が詰まる様子」を見せてくれる経路とは限らない --
より寛容なこのオラクルでは、その経路自体は個別に問題なく通れる
一方で、全体のグリッド探索ではstartからgoalへの接続が完全に
見つからない(別の場所にあるボトルネックがすべての経路を塞いで
いる)ということが起こり得る。そこで2つのパネルは、独立に見つけた
2本の経路を表示する: 箱単体プランナーの完全な解と、運搬者オラクル
の*best-effort*経路(track_best_effort=Trueでのgrid_bfs)、
つまり隣接セルがすべてブロックされる直前まで到達できた
一番ゴールに近い状態。この止まる場所は探索グラフ上の本物の
行き止まりであり、演出でその場に揺らしているわけではない。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import phase1_demo as m


def interpolate_path(path, sub_steps=4):
    """grid_bfsの粗い経路(グリッドセル1つにつき1点なので、フレーム間で
    見た目上「飛ぶ」)を、連続する状態のペアごとに位置と角度を
    sub_steps回線形補間して滑らかにする。angle_wrapによって、
    回転が-pi/+piの継ぎ目を通って「遠回り」しないようにしている。
    """
    dense = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        for i in range(1, sub_steps + 1):
            t = i / sub_steps
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            dth = m.angle_wrap(b[2] - a[2])
            th = m.angle_wrap(a[2] + dth * t)
            dense.append((x, y, th))
    return dense


def draw_carrier(ax, x, y, theta, color='tab:orange', alpha=0.5, num_carriers=None):
    """すべての運搬者カプセル(num_carriersに応じて1個か2個)と、
    向きが分かるようにするための線。"""
    r = m.HUMAN_R
    for cx, cy in m.best_human_positions(x, y, theta, num_carriers):
        ax.add_patch(patches.Circle((cx, cy), r, facecolor=color, edgecolor='k', alpha=alpha, zorder=1))
        nose = np.array([cx, cy]) + r * 0.9 * np.array([np.cos(theta), np.sin(theta)])
        ax.plot([cx, nose[0]], [cy, nose[1]], color='k', linewidth=2, alpha=min(1.0, alpha + 0.3), zorder=2)


def build_gif(out_path="phase1_demo.gif", fps=12, sub_steps=6, hold_frames=20, num_carriers=None):
    """2パネル比較GIFを描画してout_pathに保存する。

    左パネル: 箱単体の経路(ここでは常にゴールに到達する -- なぜ
    これが「簡単な」ケースなのかはモジュールdocstring参照)。
    右パネル: 運搬者オラクルのbest-effort経路。探索がブロックされて
    いない隣接セルを使い果たす直前の、一番ゴールに近かった姿勢で
    止まる(演出ではなく本物の行き止まり)。GIFがループする前に
    最終的なPASS/BLOCKED状態を読み取りやすいよう、両パネルとも
    最後のフレームでhold_frames分だけ静止する。
    """
    path_box, _, _, _, _ = m.grid_bfs(m.START, m.GOAL, with_human=False)
    assert path_box is not None, "box-only path must exist"

    full_path, best_path, _, _, _, _ = m.grid_bfs(
        m.START, m.GOAL, with_human=True, track_best_effort=True, num_carriers=num_carriers)
    assert full_path is None, "expected the carrier case to be blocked for this demo"
    assert best_path is not None and len(best_path) > 1

    dense_box = interpolate_path(path_box, sub_steps=sub_steps)
    dense_carrier = interpolate_path(best_path, sub_steps=sub_steps)
    n_box, n_carrier = len(dense_box), len(dense_carrier)
    # 2本の経路は長さが違い(片方はゴールに到達し、片方はしない)、
    # それぞれ独立した時間軸で表示される: ループしたりリセット
    # したりせず、尽きたらそのパネルは最後のフレームで止まるだけ。
    total_frames = max(n_box, n_carrier) + hold_frames

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    def render(frame_i):
        for ax in axes:
            ax.clear()  # matplotlibには「フレームNで再描画」という組み込み機能がないので、毎回全部描き直すのが一番簡単

        left_i = min(frame_i, n_box - 1)
        left_done = frame_i >= n_box - 1
        m.draw_env(axes[0], "Box only" + ("  -- REACHED GOAL" if left_done else ""))
        m.draw_box(axes[0], *dense_box[left_i], color='tab:green' if left_done else 'tab:blue', alpha=0.9)

        right_i = min(frame_i, n_carrier - 1)
        stuck = frame_i >= n_carrier - 1
        m.draw_env(axes[1], "Box + carrier" + ("  -- BLOCKED: no path reaches the goal" if stuck else ""))
        box_color = 'tab:red' if stuck else 'tab:blue'
        carrier_color = 'tab:red' if stuck else 'tab:orange'
        m.draw_box(axes[1], *dense_carrier[right_i], color=box_color, alpha=0.9)
        draw_carrier(axes[1], *dense_carrier[right_i], color=carrier_color, alpha=0.65 if stuck else 0.5,
                     num_carriers=num_carriers)

        fig.suptitle("Odoriba Phase 1: box-only (complete) vs. box+carrier (best-effort) "
                      f"(corridor {m.W1:.0f}x{m.W2:.0f}cm, box {m.BOX_L:.0f}x{m.BOX_W:.0f}cm)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    anim = animation.FuncAnimation(fig, render, frames=total_frames, interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"box-only path: {n_box} frames (reaches goal)")
    print(f"carrier best-effort path: {n_carrier} frames (does not reach goal)")
    print(f"saved gif to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carriers", type=int, choices=(1, 2), default=m.NUM_CARRIERS,
        help="How many people carry the box: 1 (solo, trailing behind) "
             "or 2 (one at each end). Default: %(default)s.")
    args = parser.parse_args()
    build_gif(num_carriers=args.carriers)
