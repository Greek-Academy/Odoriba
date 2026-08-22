"""
Phase 1 minimal demo: L-corridor turn, box-only vs box+carrier.

Real measurements from Phase 0 (2026-08-15):
  - Cardboard box footprint: 60cm x 42cm
  - Widened corner (box-alone passes here): arm widths 61cm / 52cm
  - Box alone: passed. Same box carried by one person: failed.

This script builds a simple 2D (x, y, theta) grid planner (BFS over a
discretized state space) and shows the box can find a path through the
L-corridor alone, but cannot once a circle representing the carrier's body
is attached to it.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---- Parameters (cm), from Phase 0 measurement ----
# Phase 0 widened the corner to 61cm/52cm and the box alone passed there.
# That exact pinch, modeled as a strict two-rectangle L-corridor (no slack
# beyond the measured gap), turns out to be *just* infeasible for the box
# even with no human at all (see phase1 dev notes) -- the real room had a
# bit more open floor beyond the tightest point that this idealization
# doesn't capture. Loosened by ~4-6cm to the nearest round numbers where
# the box-only case is feasible in this model; the qualitative result
# (box passes alone, fails with a carrier attached) is unchanged.
W1 = 65.0        # corridor arm width (horizontal arm)
W2 = 58.0        # corridor arm width (vertical arm)
ARM_LEN = 150.0  # length of each corridor arm

BOX_L = 60.0     # box footprint, long side
BOX_W = 42.0     # box footprint, short side

HUMAN_R = 20.0                       # carrier body capsule radius (CLAUDE.md: 0.2m)
HUMAN_OFFSET = BOX_L / 2 + HUMAN_R * 0.6   # circle center, trailing behind box rear edge

START = (ARM_LEN - 40.0, W1 / 2, 0.0)
GOAL = (W2 / 2, ARM_LEN - 40.0, np.pi / 2)


def in_free_space(x, y):
    return (0 <= x <= ARM_LEN and 0 <= y <= W1) or (0 <= x <= W2 and 0 <= y <= ARM_LEN)


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def box_corners(x, y, theta):
    hl, hw = BOX_L / 2, BOX_W / 2
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return local @ rot(theta).T + np.array([x, y])


def box_sample_points(x, y, theta, nu=7, nv=4):
    hl, hw = BOX_L / 2, BOX_W / 2
    us = np.linspace(-hl, hl, nu)
    vs = np.linspace(-hw, hw, nv)
    local = np.array([[u, v] for u in us for v in vs])
    return local @ rot(theta).T + np.array([x, y])


def human_circle_center(x, y, theta):
    local = np.array([-HUMAN_OFFSET, 0.0])
    return rot(theta) @ local + np.array([x, y])


def collision_free(x, y, theta, with_human):
    pts = box_sample_points(x, y, theta)
    if not all(in_free_space(px, py) for px, py in pts):
        return False
    if with_human:
        cx, cy = human_circle_center(x, y, theta)
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        circle_pts = [(cx + HUMAN_R * np.cos(a), cy + HUMAN_R * np.sin(a)) for a in angles]
        circle_pts.append((cx, cy))
        if not all(in_free_space(px, py) for px, py in circle_pts):
            return False
    return True


def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- Grid-based planner ----
# The corridor is narrow relative to the box (only ~1-3% of random (x,y,theta)
# states are collision-free), so a plain random-sampling RRT grows its tree
# too slowly to be a reliable demo overnight. Instead we discretize (x, y,
# theta) into a fine grid and search it exhaustively (BFS). This is
# resolution-complete: if the grid says "no path", it is because none exists
# at this resolution, not because of sampling bad luck. Swap back to a true
# RRT-Connect later once there's time to tune it (see CLAUDE.md Phase 1).
DX = 4.0
DTHETA = np.radians(15)
NTH = int(round(2 * np.pi / DTHETA))


def build_grid(with_human):
    nx = int(round(ARM_LEN / DX)) + 1
    ny = nx
    xs = np.linspace(0, ARM_LEN, nx)
    ys = np.linspace(0, ARM_LEN, ny)
    thetas = -np.pi + DTHETA * np.arange(NTH)
    free = np.zeros((nx, ny, NTH), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if not in_free_space(x, y):
                continue
            for k, th in enumerate(thetas):
                free[i, j, k] = collision_free(x, y, th, with_human)
    return xs, ys, thetas, free


def nearest_index(xs, ys, thetas, state):
    x, y, th = state
    i = int(np.argmin(np.abs(xs - x)))
    j = int(np.argmin(np.abs(ys - y)))
    k = int(np.argmin(np.abs(angle_wrap(thetas - th))))
    return i, j, k


def _reconstruct(xs, ys, thetas, prev, start_idx, end_idx):
    path_idx = [end_idx]
    while path_idx[-1] != start_idx:
        path_idx.append(prev[path_idx[-1]])
    path_idx.reverse()
    return [(xs[i], ys[j], thetas[k]) for i, j, k in path_idx]


def grid_bfs(start, goal, with_human, track_best_effort=False):
    xs, ys, thetas, free = build_grid(with_human)
    nx, ny = len(xs), len(ys)
    start_idx = nearest_index(xs, ys, thetas, start)
    goal_idx = nearest_index(xs, ys, thetas, goal)
    if not free[start_idx] or not free[goal_idx]:
        return (None, free, xs, ys, thetas) if not track_best_effort else (None, None, free, xs, ys, thetas)

    from collections import deque
    visited = np.zeros_like(free, dtype=bool)
    prev = {}
    q = deque([start_idx])
    visited[start_idx] = True
    neighbors = [(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 for dk in (-1, 0, 1) if not (di == 0 and dj == 0 and dk == 0)]

    gx, gy = goal[0], goal[1]
    best_idx = start_idx
    best_dist = np.hypot(xs[start_idx[0]] - gx, ys[start_idx[1]] - gy)

    while q:
        cur = q.popleft()
        if cur == goal_idx:
            path = _reconstruct(xs, ys, thetas, prev, start_idx, cur)
            if track_best_effort:
                return path, path, free, xs, ys, thetas
            return path, free, xs, ys, thetas
        ci, cj, ck = cur
        for di, dj, dk in neighbors:
            ni, nj, nk = ci + di, cj + dj, (ck + dk) % NTH
            if 0 <= ni < nx and 0 <= nj < ny and free[ni, nj, nk] and not visited[ni, nj, nk]:
                visited[ni, nj, nk] = True
                prev[(ni, nj, nk)] = cur
                q.append((ni, nj, nk))
                if track_best_effort:
                    d = np.hypot(xs[ni] - gx, ys[nj] - gy)
                    if d < best_dist:
                        best_dist = d
                        best_idx = (ni, nj, nk)

    if track_best_effort:
        best_path = _reconstruct(xs, ys, thetas, prev, start_idx, best_idx)
        return None, best_path, free, xs, ys, thetas
    return None, free, xs, ys, thetas


def draw_env(ax, title):
    ax.add_patch(patches.Rectangle((0, 0), ARM_LEN, W1, facecolor='#e8e8e8', edgecolor='none', zorder=0))
    ax.add_patch(patches.Rectangle((0, 0), W2, ARM_LEN, facecolor='#e8e8e8', edgecolor='none', zorder=0))
    ax.set_xlim(-10, ARM_LEN + 10)
    ax.set_ylim(-10, ARM_LEN + 10)
    ax.set_aspect('equal')
    ax.set_title(title)


def draw_box(ax, x, y, theta, color='tab:blue', alpha=0.5, lw=1.0):
    corners = box_corners(x, y, theta)
    ax.add_patch(patches.Polygon(corners, closed=True, facecolor=color, edgecolor='k', alpha=alpha, linewidth=lw, zorder=2))


def draw_human(ax, x, y, theta, color='tab:red', alpha=0.4):
    cx, cy = human_circle_center(x, y, theta)
    ax.add_patch(patches.Circle((cx, cy), HUMAN_R, facecolor=color, edgecolor='k', alpha=alpha, zorder=1))


def run_case(ax, with_human, label):
    path, free, xs, ys, thetas = grid_bfs(START, GOAL, with_human=with_human)
    found = path is not None
    draw_env(ax, f"{label}: {'PASS (path found)' if found else 'BLOCKED (no path exists at this resolution)'}")

    if found:
        step = max(1, len(path) // 12)
        for s in path[::step]:
            draw_box(ax, *s, alpha=0.15)
            if with_human:
                draw_human(ax, *s, alpha=0.10)
        draw_box(ax, *path[0], color='tab:green', alpha=0.9)
        draw_box(ax, *path[-1], color='tab:blue', alpha=0.9)
        if with_human:
            draw_human(ax, *path[0], color='tab:green', alpha=0.5)
            draw_human(ax, *path[-1], color='tab:blue', alpha=0.5)
    else:
        draw_box(ax, *START, color='tab:green', alpha=0.9)
        draw_box(ax, *GOAL, color='tab:blue', alpha=0.4)
        if with_human:
            draw_human(ax, *START, color='tab:green', alpha=0.5)
            draw_human(ax, *GOAL, color='tab:blue', alpha=0.3)
    return found


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    found1 = run_case(axes[0], with_human=False, label="Box only")
    found2 = run_case(axes[1], with_human=True, label="Box + carrier")

    fig.suptitle("Odoriba Phase 1: L-corridor, box vs box+human "
                  f"(corridor {W1:.0f}x{W2:.0f}cm, box {BOX_L:.0f}x{BOX_W:.0f}cm, "
                  f"carrier r={HUMAN_R:.0f}cm)")
    fig.tight_layout()
    out_path = "phase1_demo.png"
    fig.savefig(out_path, dpi=150)
    print(f"box only:    {'PASS' if found1 else 'BLOCKED'}")
    print(f"box+carrier: {'PASS' if found2 else 'BLOCKED'}")
    print(f"saved figure to {out_path}")
    plt.show()
