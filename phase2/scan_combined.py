"""
実スキャン同士の組み合わせデモ: スキャンした家具(段ボール)を、スキャン
した実際の階段の中で動かして見せる。

    python scan_combined.py <stair.obj> <box.obj>            # 3D HTML
    python scan_combined.py <stair.obj> <box.obj> --out x.html

いまの割り切り(正直な注意):
  階段スキャンは開いた面で水密化できないため、厳密な「通る/通らない」は
  出せない。ここでは家具を階段の中心線に沿って動かす『可視化』と、家具
  表面から階段面までの最小距離(符号なし=近さの目安)の表示までにとどめる。
  厳密な合否は、より完全な階段スキャン(水密化)か実測寸法の箱モデルが必要。
"""

import argparse

import numpy as np
import trimesh

import phase2_demo as p
import scan_demo as sd
import scan_furniture as sf
import mesh_env as me


def vertex_colors(mesh):
    """メッシュの頂点色(N,3 uint8)を返す。テクスチャがあればそれをサンプル、
    無ければNone。Scaniverseのスキャンはテクスチャjpgを持つ。"""
    try:
        col = np.asarray(mesh.visual.to_color().vertex_colors)
        if col is not None and len(col) == len(mesh.vertices):
            return col[:, :3].astype(int)
    except Exception:
        pass
    return None


def _rgb_strings(colors):
    """(N,3)の色配列を plotly 用の 'rgb(r,g,b)' 文字列リストにする。"""
    return ["rgb(%d,%d,%d)" % (r, g, b) for r, g, b in colors]


def isolate_box_colored(mesh):
    """箱を切り出して(点群, 頂点色)を返す。scan_furnitureの切り出しと同じ
    「縦に厚みのあるセル=箱」判定を、頂点色も一緒に持って行う。"""
    from scipy import ndimage
    v = np.asarray(mesh.vertices)
    col = vertex_colors(mesh)
    xy_bin = 5.0
    bx = np.arange(v[:, 0].min(), v[:, 0].max() + xy_bin, xy_bin)
    by = np.arange(v[:, 1].min(), v[:, 1].max() + xy_bin, xy_bin)
    ix = np.clip(np.digitize(v[:, 0], bx) - 1, 0, len(bx) - 2)
    iy = np.clip(np.digitize(v[:, 1], by) - 1, 0, len(by) - 2)
    zmax = np.full((len(bx) - 1, len(by) - 1), -1e9)
    zmin = np.full((len(bx) - 1, len(by) - 1), 1e9)
    np.maximum.at(zmax, (ix, iy), v[:, 2])
    np.minimum.at(zmin, (ix, iy), v[:, 2])
    zrange = np.where(zmax > -1e8, zmax - zmin, 0.0)
    lbl, n = ndimage.label(zrange > 20.0)
    if n == 0:
        return v, col
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    xi, yi = np.where(lbl == sizes.argmax())
    x0, x1 = bx[xi.min()], bx[xi.max() + 1]
    y0, y1 = by[yi.min()], by[yi.max() + 1]
    m = 3.0
    sel = ((v[:, 0] >= x0 - m) & (v[:, 0] <= x1 + m) &
           (v[:, 1] >= y0 - m) & (v[:, 1] <= y1 + m))
    return v[sel], (col[sel] if col is not None else None)


def box_poses_along_stair(stair_mesh, box_dims, n=7):
    """階段の中心線に沿って、家具を置く姿勢を並べる。
    向きは中心線の接線(ヨー)と勾配(ピッチ)に合わせる。"""
    line = sd.centerline(stair_mesh)
    idx = np.linspace(0, len(line) - 1, n).round().astype(int)
    poses = []
    for i in idx:
        pos = line[i].copy()
        pos[2] += box_dims[2] / 2 + 5.0  # 歩行面の少し上に浮かせる
        a = line[max(i - 1, 0)]
        b = line[min(i + 1, len(line) - 1)]
        d = b - a
        yaw = np.degrees(np.arctan2(d[1], d[0]))
        horiz = np.hypot(d[0], d[1])
        pitch = np.degrees(np.arctan2(d[2], horiz)) if horiz > 1e-6 else 0.0
        # 傾きは運搬の常識的上限まで
        import phase2_lstair as L
        poses.append((pos, L._pose(yaw, min(pitch, L.MAX_TILT_DEG)).as_quat()))
    return poses


def render_combined(stair_mesh, stair_col, box_dims, poses, out_path):
    """階段を面(サーフェス)で、家具を明るいソリッドな箱で描く3D HTML。
    点群ではなくメッシュの面で描くので見た目が分かりやすい(Babylon.js相当の
    面表示をplotlyのMesh3dで実現)。家具は姿勢を変えながら階段の中を動く。"""
    import plotly.graph_objects as go
    import phase2_lstair_3d as v3

    v = np.asarray(stair_mesh.vertices)
    f = np.asarray(stair_mesh.faces)
    # 階段: メッシュの面をそのまま(実際の色を頂点色に)。点ではなく面なので密。
    stair_kw = {}
    if stair_col is not None:
        stair_kw["vertexcolor"] = stair_col.astype(np.uint8)
    else:
        stair_kw["color"] = "#9aa1ab"
    stair_trace = go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                            i=f[:, 0], j=f[:, 1], k=f[:, 2],
                            opacity=1.0, lighting=dict(ambient=0.6, diffuse=0.7),
                            flatshading=True, name="stair", showscale=False,
                            hoverinfo="skip", **stair_kw)

    half = np.asarray(box_dims, dtype=float) / 2.0

    def box_trace(pose):
        pos, quat = pose
        # 明るいオレンジのソリッドな箱。面で描くので埋もれず分かりやすい
        return v3.box_mesh(np.asarray(pos), p.g3.rotmat_from_quat(quat), half,
                           "#ff7a1a", opacity=1.0)

    fig = go.Figure()
    fig.add_trace(stair_trace)
    fig.add_trace(box_trace(poses[0]))
    dyn = [len(fig.data) - 1]
    fig.frames = [go.Frame(data=[box_trace(ps)], traces=dyn, name=str(k))
                  for k, ps in enumerate(poses)]

    steps = [dict(method="animate", label=f"{k+1}",
                  args=[[str(k)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for k in range(len(poses))]
    d = np.asarray(box_dims, dtype=float)
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=500, redraw=True),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)])])],
        sliders=[dict(steps=steps, x=0.12, len=0.85, y=0.04)],
        title=dict(text=f"Scanned cardboard ({d[0]:.0f}x{d[1]:.0f}x{d[2]:.0f}cm) "
                        "moving through the REAL scanned staircase -- drag to rotate", x=0.5),
        scene=dict(aspectmode="data", camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0)),
                   xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        paper_bgcolor="#0e1116", font=dict(color="#dddddd"),
        margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(out_path, include_plotlyjs=True, auto_play=False)
    print(f"saved combined 3D to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stair", help="階段のスキャン OBJ/PLY")
    parser.add_argument("box", help="家具のスキャン OBJ/PLY")
    parser.add_argument("--n", type=int, default=7, help="並べる姿勢の数")
    parser.add_argument("--out", default="scan_combined.html")
    args = parser.parse_args()

    stair_mesh, _ = sd.load_scan_zup(args.stair)
    stair_col = vertex_colors(stair_mesh)
    box_mesh, _ = sd.load_scan_zup(args.box)
    box_pts, _ = isolate_box_colored(box_mesh)
    # 家具の外形寸法(OBB)。表示は寸法のソリッド箱にするので点群は使わない
    _, ext = trimesh.bounds.oriented_bounds(box_pts)
    dims = np.sort(ext)[::-1]

    print(f"階段スキャン: 面{len(stair_mesh.faces)} (色{'あり' if stair_col is not None else 'なし'}) "
          f"/ 家具dims= {dims.round(0)} cm")
    poses = box_poses_along_stair(stair_mesh, dims, n=args.n)
    render_combined(stair_mesh, stair_col, dims, poses, args.out)
