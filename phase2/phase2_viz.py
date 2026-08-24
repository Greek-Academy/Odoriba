"""
Phase 2 の可視化: 階段を横(y-z平面、側面図)から見た図。

Phase 1のdraw_env/draw_box/draw_carriersと同じ役割だが、3Dなので
そのまま2D描画はできない。x方向(階段の幅)はこの主張には効かない
(天井高・踏み外しはy-z平面の話)ため、側面図に投影して描く。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

import phase2_demo as p


def _order_ccw(points_2d):
    """2D点群の凸包の頂点を順序付きで返す。

    箱の表面サンプル点(面ごとに何十点もある)をそのまま重心角度で
    並べても、内部寄りの点と外周の点が混ざって星形にジグザグする
    ため、実際に凸包(外周の頂点だけ)を計算してから使う。
    """
    pts = np.asarray(points_2d)
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def draw_env_side(ax, outer, obstacles, title):
    """階段室を側面図(y-z)で描く: 各ステップを灰色の矩形として。"""
    outer_center, _, outer_half = outer
    total_y, total_z = outer_half[1] * 2, outer_half[2] * 2
    for center, _, half in obstacles:
        y0, z0 = center[1] - half[1], center[2] - half[2]
        ax.add_patch(patches.Rectangle((y0, z0), half[1] * 2, half[2] * 2,
                                        facecolor="#c8c8c8", edgecolor="none", zorder=0))
    ax.set_xlim(-20, total_y + 20)
    ax.set_ylim(-10, total_z + 10)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("y (cm, direction of travel)")
    ax.set_ylabel("z (cm, height)")


def draw_furniture_side(ax, pos, quat, color="tab:blue", alpha=0.6):
    pts3d = p.furniture_world_points(pos, quat)
    pts2d = _order_ccw(pts3d[:, [1, 2]])
    ax.add_patch(patches.Polygon(pts2d, closed=True, facecolor=color,
                                  edgecolor="k", alpha=alpha, zorder=2))


def draw_carriers_side(ax, pos, quat, outer, obstacles, num_carriers=None, color="tab:red", alpha=0.5):
    for c in p.best_human_positions(pos, quat, outer, obstacles, num_carriers):
        y0, z0 = c[1] - p.HUMAN_R, c[2] - p.HUMAN_HEIGHT / 2
        ax.add_patch(patches.Rectangle((y0, z0), p.HUMAN_R * 2, p.HUMAN_HEIGHT,
                                        facecolor=color, edgecolor="k", alpha=alpha, zorder=1))


def draw_path_side(ax, path, outer, obstacles, with_human, num_carriers=None,
                    ghost_n=10, worst_idx=None):
    """経路の残像(薄い箱)と、始点・終点(濃い箱)を描く。
    worst_idxが指定されていれば、その位置を赤で強調する(詰まる箇所)。
    """
    step = max(1, len(path) // ghost_n)
    for i, (pos, q) in enumerate(path[::step]):
        draw_furniture_side(ax, pos, q, color="tab:blue", alpha=0.12)
        if with_human:
            draw_carriers_side(ax, pos, q, outer, obstacles, num_carriers, alpha=0.08)
    draw_furniture_side(ax, *path[0], color="tab:green", alpha=0.85)
    draw_furniture_side(ax, *path[-1], color="tab:blue", alpha=0.85)
    if with_human:
        draw_carriers_side(ax, *path[0], outer, obstacles, num_carriers, color="tab:green", alpha=0.5)
        draw_carriers_side(ax, *path[-1], outer, obstacles, num_carriers, color="tab:blue", alpha=0.5)
    if worst_idx is not None:
        pos, q = path[worst_idx]
        draw_furniture_side(ax, pos, q, color="tab:red", alpha=0.9)
        if with_human:
            draw_carriers_side(ax, pos, q, outer, obstacles, num_carriers, color="tab:red", alpha=0.7)
