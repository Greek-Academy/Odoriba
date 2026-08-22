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

HUMAN_R = 20.0     # carrier body capsule radius (CLAUDE.md: 0.2m)
# Arm reach from the grip point to where the carrier's body center sits.
# Not yet measured in Phase 0 -- placeholder pending a real value, chosen
# to be the same order of magnitude as the box and corridor.
CARRY_ARM = 15.0
# Lateral stances the carrier may take relative to the box centerline.
SIDE_OFFSETS = (-15.0, -7.5, 0.0, 7.5, 15.0)

# How many people are carrying the box: 2 (one at each end -- the usual
# case for furniture) or 1 (one person, trailing behind the rear grip
# point, e.g. someone solo-carrying this cardboard box). Every oracle
# function below defaults to this global but also accepts an explicit
# num_carriers= override, the same pattern used for w1/w2/arm_len.
NUM_CARRIERS = 2

# START/GOAL sit this far from each arm's modeled end. Must clear
# BOX_L/2 + CARRY_ARM + HUMAN_R (here 30+15+20=65cm) so the start/goal
# poses themselves are carriable -- the modeled arm end is a stand-in for
# "corridor keeps going", not a real wall, and a too-small margin makes a
# trailing capsule stick out past it even though nothing is really there.
END_MARGIN = 75.0
START = (ARM_LEN - END_MARGIN, W1 / 2, 0.0)
GOAL = (W2 / 2, ARM_LEN - END_MARGIN, np.pi / 2)


def in_free_space(x, y, w1=None, w2=None, arm_len=None):
    w1 = W1 if w1 is None else w1
    w2 = W2 if w2 is None else w2
    arm_len = ARM_LEN if arm_len is None else arm_len
    return (0 <= x <= arm_len and 0 <= y <= w1) or (0 <= x <= w2 and 0 <= y <= arm_len)


def _point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return float(np.hypot(px - cx, py - cy))


def dist_to_boundary(x, y, w1=None, w2=None, arm_len=None):
    """Distance from (x, y) to the nearest wall of the L-shaped free space.

    The free space is the union of two rectangles, whose boundary is the
    hexagon (0,0) -> (arm_len,0) -> (arm_len,w1) -> (w2,w1) -> (w2,arm_len)
    -> (0,arm_len). Distance to that hexagon's edges is exactly the
    clearance to the nearest wall for any point inside it.
    """
    w1 = W1 if w1 is None else w1
    w2 = W2 if w2 is None else w2
    arm_len = ARM_LEN if arm_len is None else arm_len
    verts = [(0, 0), (arm_len, 0), (arm_len, w1), (w2, w1), (w2, arm_len), (0, arm_len)]
    n = len(verts)
    return min(_point_seg_dist(x, y, *verts[i], *verts[(i + 1) % n]) for i in range(n))


def signed_clearance(x, y, w1=None, w2=None, arm_len=None):
    """+cm of margin to the nearest wall if inside; -cm of overshoot if outside."""
    d = dist_to_boundary(x, y, w1, w2, arm_len)
    return d if in_free_space(x, y, w1, w2, arm_len) else -d


def shape_clearance(points, w1=None, w2=None, arm_len=None):
    """Worst-case (minimum) clearance over a set of boundary sample points."""
    return min(signed_clearance(px, py, w1, w2, arm_len) for px, py in points)


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


def box_clearance(x, y, theta, w1=None, w2=None, arm_len=None):
    return shape_clearance(box_sample_points(x, y, theta), w1, w2, arm_len)


def circle_boundary_points(center, radius=HUMAN_R, n=16):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]


def place_humans(x, y, theta, side_offset, num_carriers=None):
    """Carrier capsule center(s) for a given lateral stance.

    Grip points sit at the box's end(s), each extended outward by
    CARRY_ARM along the box's long axis, with a shared lateral shift
    (side_offset) perpendicular to it -- this is CLAUDE.md's
    carriable(q, h) oracle, where h ranges over SIDE_OFFSETS.

    num_carriers=2 (default): one person at each end -- returns
    (front, back).
    num_carriers=1: a single person trailing behind the box, gripping
    only the rear point -- returns (back,) as a 1-tuple. Always a
    1-tuple/2-tuple of (x, y) points, so callers can just loop over it
    regardless of which case they're in.
    """
    num_carriers = NUM_CARRIERS if num_carriers is None else num_carriers
    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-d[1], d[0]])
    c = np.array([x, y])
    hl = BOX_L / 2

    grip_back = c - d * hl
    h_back = tuple(grip_back - d * CARRY_ARM + n * side_offset)
    if num_carriers == 1:
        return (h_back,)

    grip_front = c + d * hl
    h_front = tuple(grip_front + d * CARRY_ARM + n * side_offset)
    return (h_front, h_back)


def carriable_clearance(x, y, theta, w1=None, w2=None, arm_len=None, num_carriers=None):
    """Best clearance achievable over all allowed carrier stances (cm).

    Tries every stance in SIDE_OFFSETS and keeps the most favorable one,
    i.e. this answers "does *some* way of standing next to the box work,
    and if so by how much clearance" -- not "does this one exact stance
    work". Returns (clearance_cm, best_side_offset); best_side_offset is
    None if the box itself does not fit (carrier placement is then moot).
    """
    bc = box_clearance(x, y, theta, w1, w2, arm_len)
    if bc < 0:
        return bc, None
    best = -np.inf
    best_offset = None
    for off in SIDE_OFFSETS:
        capsule_centers = place_humans(x, y, theta, off, num_carriers)
        capsule_clearances = [
            shape_clearance(circle_boundary_points(p), w1, w2, arm_len)
            for p in capsule_centers
        ]
        # The whole arrangement (box + every carrier) is only as good as
        # its tightest point.
        cand = min([bc] + capsule_clearances)
        if cand > best:
            best = cand
            best_offset = off
    return best, best_offset


def best_human_positions(x, y, theta, num_carriers=None):
    """Carrier capsule center(s) at the best offset, for drawing only."""
    _, offset = carriable_clearance(x, y, theta, num_carriers=num_carriers)
    if offset is None:
        offset = 0.0  # box itself already fails; offset choice is moot, just draw *something*
    return place_humans(x, y, theta, offset, num_carriers)


def collision_free(x, y, theta, with_human, w1=None, w2=None, arm_len=None, num_carriers=None):
    if with_human:
        clearance, _ = carriable_clearance(x, y, theta, w1, w2, arm_len, num_carriers)
        return clearance >= 0
    return box_clearance(x, y, theta, w1, w2, arm_len) >= 0


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


def build_grid(with_human, w1=None, w2=None, arm_len=None, num_carriers=None):
    arm_len = ARM_LEN if arm_len is None else arm_len
    nx = int(round(arm_len / DX)) + 1
    ny = nx
    xs = np.linspace(0, arm_len, nx)
    ys = np.linspace(0, arm_len, ny)
    thetas = -np.pi + DTHETA * np.arange(NTH)
    free = np.zeros((nx, ny, NTH), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if not in_free_space(x, y, w1, w2, arm_len):
                continue
            for k, th in enumerate(thetas):
                free[i, j, k] = collision_free(x, y, th, with_human, w1, w2, arm_len, num_carriers)
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


def grid_bfs(start, goal, with_human, track_best_effort=False, w1=None, w2=None, arm_len=None, num_carriers=None):
    xs, ys, thetas, free = build_grid(with_human, w1, w2, arm_len, num_carriers)
    nx, ny = len(xs), len(ys)
    start_idx = nearest_index(xs, ys, thetas, start)
    goal_idx = nearest_index(xs, ys, thetas, goal)
    if not free[start_idx]:
        return (None, free, xs, ys, thetas) if not track_best_effort else (None, None, free, xs, ys, thetas)
    if not free[goal_idx] and not track_best_effort:
        # No exact path can end on a blocked goal cell; only worth the full
        # search below when the caller wants the best-effort approach instead.
        return None, free, xs, ys, thetas

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


def path_min_clearance(path, with_human, num_carriers=None):
    """Worst (tightest) clearance encountered along a found path, in cm."""
    if with_human:
        return min(carriable_clearance(*s, num_carriers=num_carriers)[0] for s in path)
    return min(box_clearance(*s) for s in path)


def find_critical_width(lo=40.0, hi=70.0, iters=10, num_carriers=None):
    """Binary search a single (W1=W2=w) corridor width where the box-only
    planner still finds a path but the carriability oracle does not.

    Temporarily overrides the module-level W1/W2 globals for each trial
    (build_grid/in_free_space fall back to them when w1/w2 aren't passed
    explicitly), restoring the originals afterward.
    """
    global W1, W2
    orig_w1, orig_w2 = W1, W2
    results = []
    found_w = None
    try:
        for _ in range(iters):
            w = (lo + hi) / 2.0
            start = (ARM_LEN - END_MARGIN, w / 2.0, 0.0)
            goal = (w / 2.0, ARM_LEN - END_MARGIN, np.pi / 2)
            W1, W2 = w, w
            path_free, *_ = grid_bfs(start, goal, with_human=False)
            path_human, *_ = grid_bfs(start, goal, with_human=True, num_carriers=num_carriers)
            ok_free = path_free is not None
            ok_human = path_human is not None
            results.append((w, ok_free, ok_human))
            if ok_free and not ok_human:
                found_w = w
                break
            elif not ok_free:
                lo = w
            else:
                hi = w
    finally:
        W1, W2 = orig_w1, orig_w2
    return found_w, results


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


def draw_carriers(ax, x, y, theta, color='tab:red', alpha=0.4, num_carriers=None):
    """Every carrier capsule (1 or 2, per num_carriers) at the best side offset."""
    for c in best_human_positions(x, y, theta, num_carriers):
        ax.add_patch(patches.Circle(c, HUMAN_R, facecolor=color, edgecolor='k', alpha=alpha, zorder=1))


def run_case(ax, with_human, label, num_carriers=None):
    path, free, xs, ys, thetas = grid_bfs(START, GOAL, with_human=with_human, num_carriers=num_carriers)
    found = path is not None
    draw_env(ax, f"{label}: {'PASS (path found)' if found else 'BLOCKED (no path exists at this resolution)'}")

    if found:
        step = max(1, len(path) // 12)
        for s in path[::step]:
            draw_box(ax, *s, alpha=0.15)
            if with_human:
                draw_carriers(ax, *s, alpha=0.10, num_carriers=num_carriers)
        draw_box(ax, *path[0], color='tab:green', alpha=0.9)
        draw_box(ax, *path[-1], color='tab:blue', alpha=0.9)
        if with_human:
            draw_carriers(ax, *path[0], color='tab:green', alpha=0.5, num_carriers=num_carriers)
            draw_carriers(ax, *path[-1], color='tab:blue', alpha=0.5, num_carriers=num_carriers)
    else:
        draw_box(ax, *START, color='tab:green', alpha=0.9)
        draw_box(ax, *GOAL, color='tab:blue', alpha=0.4)
        if with_human:
            draw_carriers(ax, *START, color='tab:green', alpha=0.5, num_carriers=num_carriers)
            draw_carriers(ax, *GOAL, color='tab:blue', alpha=0.3, num_carriers=num_carriers)
    return found, path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carriers", type=int, choices=(1, 2), default=NUM_CARRIERS,
        help="How many people carry the box: 1 (solo, trailing behind) "
             "or 2 (one at each end). Default: %(default)s.")
    args = parser.parse_args()
    num_carriers = args.carriers
    carrier_label = "1 carrier" if num_carriers == 1 else "2 carriers"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    found1, path1 = run_case(axes[0], with_human=False, label="Box only")
    found2, path2 = run_case(axes[1], with_human=True, label=f"Box + {carrier_label}", num_carriers=num_carriers)

    fig.suptitle("Odoriba Phase 1: L-corridor, box vs box+human "
                  f"(corridor {W1:.0f}x{W2:.0f}cm, box {BOX_L:.0f}x{BOX_W:.0f}cm, "
                  f"carrier r={HUMAN_R:.0f}cm, {carrier_label})")
    fig.tight_layout()
    out_path = "phase1_demo.png"
    fig.savefig(out_path, dpi=150)

    print(f"carriers: {num_carriers}")
    print(f"box only:    {'PASS' if found1 else 'BLOCKED'}"
          + (f" (min clearance {path_min_clearance(path1, False):.1f}cm)" if found1 else ""))
    print(f"box+carrier: {'PASS' if found2 else 'BLOCKED'}"
          + (f" (min clearance {path_min_clearance(path2, True, num_carriers):.1f}cm)" if found2 else ""))
    print(f"saved figure to {out_path}")

    print("\nsearching for the critical corridor width (box passes, carrier blocked)...")
    w, results = find_critical_width(num_carriers=num_carriers)
    if w is not None:
        print(f"廊下幅 {w:.1f}cm / 箱 {BOX_L:.0f}x{BOX_W:.0f}cm / 人 r={HUMAN_R:.0f}cm, arm={CARRY_ARM:.0f}cm, {carrier_label}")
        print("  自由剛体  : 通る")
        print("  人あり    : 通らない")
    else:
        print("no width in the search range separated the two cases; see `results` for the raw sweep")
    for w_, ok_free, ok_human in results:
        print(f"  w={w_:5.1f}cm  box_only={'PASS' if ok_free else 'BLOCK'}  "
              f"with_carrier={'PASS' if ok_human else 'BLOCK'}")

    plt.show()
