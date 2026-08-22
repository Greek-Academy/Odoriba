"""
Phase 1 GIF: box alone (reaches the goal) vs. box + carrier (best-effort,
does not reach the goal).

With the two-capsule + side-offset carriability oracle, the box-only
path is not guaranteed to be the trajectory that "shows" the carrier
jamming -- the more permissive oracle may find that path individually
fine while a full grid search still can't connect start to goal at all
(a different bottleneck elsewhere blocks every route). So the two panels
now show two independently-found paths: the box-only planner's complete
solution, and the carrier oracle's *best-effort* path (grid_bfs with
track_best_effort=True), i.e. the closest-to-goal state it could reach
before every neighboring cell was blocked. That stopping point is a real
dead end in the search graph, not a scripted wiggle-in-place.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import phase1_demo as m


def interpolate_path(path, sub_steps=4):
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
    """Every carrier capsule (1 or 2, per num_carriers) plus a facing marker."""
    r = m.HUMAN_R
    for cx, cy in m.best_human_positions(x, y, theta, num_carriers):
        ax.add_patch(patches.Circle((cx, cy), r, facecolor=color, edgecolor='k', alpha=alpha, zorder=1))
        nose = np.array([cx, cy]) + r * 0.9 * np.array([np.cos(theta), np.sin(theta)])
        ax.plot([cx, nose[0]], [cy, nose[1]], color='k', linewidth=2, alpha=min(1.0, alpha + 0.3), zorder=2)


def build_gif(out_path="phase1_demo.gif", fps=12, sub_steps=6, hold_frames=20, num_carriers=None):
    path_box, _, _, _, _ = m.grid_bfs(m.START, m.GOAL, with_human=False)
    assert path_box is not None, "box-only path must exist"

    full_path, best_path, _, _, _, _ = m.grid_bfs(
        m.START, m.GOAL, with_human=True, track_best_effort=True, num_carriers=num_carriers)
    assert full_path is None, "expected the carrier case to be blocked for this demo"
    assert best_path is not None and len(best_path) > 1

    dense_box = interpolate_path(path_box, sub_steps=sub_steps)
    dense_carrier = interpolate_path(best_path, sub_steps=sub_steps)
    n_box, n_carrier = len(dense_box), len(dense_carrier)
    total_frames = max(n_box, n_carrier) + hold_frames

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    def render(frame_i):
        for ax in axes:
            ax.clear()

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
