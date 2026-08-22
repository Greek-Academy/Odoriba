"""
Phase 1 GIF: box alone vs. box + carrier on the *same* trajectory.

Earlier version let the carrier search find a path in any rotation
direction; it found one that spins the "wrong way" (through -90 degrees
instead of the +90 the box-only path uses) and drifts deep into open
corridor before running out of search budget -- so the failure didn't
visibly touch a wall and looked like a false pass.

This version keeps both panels on the exact same trajectory (the one the
box-only planner actually found). The carrier version jams very early --
before any rotation even starts -- because the person trailing behind the
box runs out of room and backs into the wall at x=0. That is a real,
visible wall contact, not an open-space stall.
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


def first_jam_index(path):
    for i, s in enumerate(path):
        if not m.collision_free(*s, with_human=True):
            return i
    return None


def draw_carrier(ax, x, y, theta, color='tab:orange', alpha=0.5):
    """Human circle plus a facing marker, so its rotation is visible."""
    cx, cy = m.human_circle_center(x, y, theta)
    r = m.HUMAN_R
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=color, edgecolor='k', alpha=alpha, zorder=1))
    nose = np.array([cx, cy]) + r * 0.9 * np.array([np.cos(theta), np.sin(theta)])
    ax.plot([cx, nose[0]], [cy, nose[1]], color='k', linewidth=2, alpha=min(1.0, alpha + 0.3), zorder=2)


def build_gif(out_path="phase1_demo.gif", fps=12, sub_steps=6, hold_frames=24, wiggle_frames=14):
    path_box, _, _, _, _ = m.grid_bfs(m.START, m.GOAL, with_human=False)
    assert path_box is not None, "box-only path must exist"

    dense = interpolate_path(path_box, sub_steps=sub_steps)
    jam_idx = first_jam_index(dense)
    assert jam_idx is not None and jam_idx > 0, "expected the carrier to jam somewhere on this path"

    jam_state = dense[jam_idx - 1]  # last state before the carrier collides
    n_frames = len(dense)
    total_frames = n_frames + hold_frames

    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    def render(frame_i):
        for ax in axes:
            ax.clear()

        left_i = min(frame_i, n_frames - 1)
        left_done = frame_i >= n_frames - 1
        m.draw_env(axes[0], "Box only" + ("  -- REACHED GOAL" if left_done else ""))
        m.draw_box(axes[0], *dense[left_i], color='tab:green' if left_done else 'tab:blue', alpha=0.9)

        jammed = frame_i >= jam_idx - 1
        if not jammed:
            right_state = dense[frame_i]
            box_color, carrier_color, carrier_alpha = 'tab:blue', 'tab:orange', 0.5
        else:
            wiggle_t = frame_i - (jam_idx - 1)
            wiggle = 1.5 * np.sin(wiggle_t * 1.3) * np.exp(-wiggle_t / 8.0) if wiggle_t < wiggle_frames else 0.0
            right_state = (jam_state[0] + wiggle, jam_state[1], jam_state[2])
            box_color, carrier_color, carrier_alpha = 'tab:red', 'tab:red', 0.65

        m.draw_env(axes[1], "Box + carrier" + ("  -- BLOCKED: carrier backs into the wall" if jammed else ""))
        m.draw_box(axes[1], *right_state, color=box_color, alpha=0.9)
        draw_carrier(axes[1], *right_state, color=carrier_color, alpha=carrier_alpha)

        fig.suptitle("Odoriba Phase 1: same path, box alone vs. box + carrier "
                      f"(corridor {m.W1:.0f}x{m.W2:.0f}cm, box {m.BOX_L:.0f}x{m.BOX_W:.0f}cm)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    anim = animation.FuncAnimation(fig, render, frames=total_frames, interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"box-only path: {n_frames} frames (reaches goal)")
    print(f"carrier jams at frame {jam_idx} / {n_frames}, state={jam_state}")
    print(f"saved gif to {out_path}")


if __name__ == "__main__":
    build_gif()
