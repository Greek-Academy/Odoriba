"""
L字階段のアニメーションGIF: 家具が階段を上り、踊り場で回る様子。
phase1_gif.py の3D版で、results/*.json から再生成する。

    python phase2_lstair_gif.py                       # results/lstair_result.json から
    python phase2_lstair_gif.py --json <path> --out <gif>

レイアウトはPNG(phase2_lstair_viz.py)と同じ2x2:
  左列 = 家具単体(ゴールに到達)、右列 = 運搬者あり(見つかった経路、
  またはbest-effort経路 -- 探索が実際に到達できた最遠の状態で止まる)。
  各列とも上段=上面図 / 下段=展開側面図。
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import geometry3d as g3
import phase2_lstair as L
import phase2_lstair_viz as viz


def interpolate_path(path, step_cm=8.0):
    """粗い経路(RRTのノード列)をdist_se3の距離でおおよそ等間隔に補間する。

    phase1_gifのinterpolate_pathと同じ役割。位置は線形、姿勢はSlerp
    (geometry3d.interpolate_se3)なので、フレーム間で姿勢が飛ばない。
    """
    dense = [path[0]]
    for (p1, q1), (p2, q2) in zip(path[:-1], path[1:]):
        d = g3.dist_se3(p1, q1, p2, q2, w=L.p.W_ROT)
        n = max(1, int(np.ceil(d / step_cm)))
        for i in range(1, n + 1):
            dense.append(g3.interpolate_se3(p1, q1, p2, q2, i / n))
    return dense


def build_gif(result, out_path, fps=12, step_cm=8.0, hold_frames=18):
    outer, obstacles = L.build_lstairs()
    meta = result["meta"]
    num_carriers = meta["carrier"]["num_carriers"]

    box = result["box_only"]
    car = result["with_carriers"]
    box_found = box["found"]
    car_found = car["found"]
    path_box = viz.parse_path(box["path"] if box_found else box["best_effort_path"])
    path_car = viz.parse_path(car["path"] if car_found else car["best_effort_path"])

    dense_box = interpolate_path(path_box, step_cm)
    dense_car = interpolate_path(path_car, step_cm)
    n_box, n_car = len(dense_box), len(dense_car)
    total = max(n_box, n_car) + hold_frames

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5),
                             gridspec_kw={"height_ratios": [1.15, 1]})
    (ax_t_box, ax_t_car), (ax_s_box, ax_s_car) = axes

    st = meta["stair"]
    fu = meta["furniture"]
    suptitle = ("Odoriba Phase 2: L-shaped staircase "
                f"(width {st['width']:.0f}cm, landing {st['landing'][0]:.0f}x"
                f"{st['landing'][1]:.0f}cm) / furniture "
                f"{fu['L']:.0f}x{fu['W']:.0f}x{fu['H']:.0f}cm")

    def render(frame_i):
        for ax in axes.ravel():
            ax.clear()  # matplotlibに部分再描画はないので毎フレーム全部描き直す

        # 左列: 家具単体
        i = min(frame_i, n_box - 1)
        done = frame_i >= n_box - 1
        title = "Furniture only" + (
            ("  -- REACHED GOAL" if box_found else "  -- stuck (best effort)")
            if done else "")
        viz.draw_env_top(ax_t_box, title)
        viz.draw_env_side(ax_s_box, "")
        color = ("tab:green" if box_found else "tab:red") if done else "tab:blue"
        viz.draw_furniture(ax_t_box, ax_s_box, *dense_box[i], color=color, alpha=0.9)

        # 右列: 運搬者あり
        j = min(frame_i, n_car - 1)
        stuck = frame_i >= n_car - 1
        title = f"Furniture + {num_carriers} carriers" + (
            ("  -- REACHED GOAL" if car_found
             else "  -- BLOCKED: search cannot get past here") if stuck else "")
        viz.draw_env_top(ax_t_car, title)
        viz.draw_env_side(ax_s_car, "")
        if car_found:
            f_color = "tab:green" if stuck else "tab:blue"
            c_color = "tab:orange"
        else:
            f_color = "tab:red" if stuck else "tab:blue"
            c_color = "tab:red" if stuck else "tab:orange"
        viz.draw_furniture(ax_t_car, ax_s_car, *dense_car[j], color=f_color, alpha=0.9)
        viz.draw_carriers(ax_t_car, ax_s_car, *dense_car[j], outer, obstacles,
                          num_carriers, color=c_color, alpha=0.65 if stuck else 0.5)

        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    anim = animation.FuncAnimation(fig, render, frames=total, interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"furniture-only: {n_box} frames ({'reaches goal' if box_found else 'best effort'})")
    print(f"with carriers: {n_car} frames ({'reaches goal' if car_found else 'does not reach goal'})")
    print(f"saved gif to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=os.path.join("results", "lstair_result.json"))
    parser.add_argument("--out", default="phase2_lstair.gif")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    build_gif(viz.load_result(args.json), args.out, fps=args.fps)
