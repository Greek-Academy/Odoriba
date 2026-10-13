"""
スキャン空間の判定結果を3D(HTML)で見せる: どこで詰まるか・何に当たるか。

  - 中心線を、その位置での余裕で色分けする(余裕があれば緑、足りなければ赤)
  - 一番狭い所での家具と運搬者(人型)を描き、足りなければ赤くする
  - 一番めり込んでいる点(当たった所)に印を付ける。手すりやドア枠に
    当たっているなら、外せば通るかをユーザーが判断できる
  - スキャンの面は、一番狭い所の周り(既定で半径2.5m)だけを半透明で描く
    (実スキャンは十万面を超えるので、全部描くとブラウザが重くなる)

    python scan_viewer.py --route                     # 検証用の合成経路(route_env)で
    python scan_viewer.py <mesh.obj>                  # 出発点・目的地は自動で提案
    python scan_viewer.py <mesh.obj> --start X Y Z --goal X Y Z [--seed X Y Z]

出力のHTMLは plotly.js を埋め込むので、ファイル1つでオフラインでも開ける。
"""

import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import phase2_lstair as L
import phase2_lstair_3d as V
import scan_plan as SP

OK_COLOR = "#2c6fbb"
NG_COLOR = "#c0392b"
CARRIER_COLOR = "#e07b39"


def _crop_mesh(mesh, center, radius):
    """center から radius 以内に重心がある三角形だけを残す。"""
    keep = np.linalg.norm(mesh.triangles_center - np.asarray(center), axis=1) <= radius
    return mesh.submesh([np.nonzero(keep)[0]], append=True)


def _mesh_trace(mesh, color="#b9c3cc", opacity=0.35):
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces)
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                     color=color, opacity=opacity, flatshading=True,
                     showlegend=False, hoverinfo="skip")


def _centerline_trace(space, sweep):
    """中心線を余裕で色分けした線。掃引していない端は灰色の点線。"""
    cl = space.centerline
    s = np.array([r[0] for r in sweep])
    c = np.array([r[1] if np.isfinite(r[1]) else np.nan for r in sweep])
    xy = np.array([cl.xy_at(v) for v in s])
    z = np.array([cl.z_at(v) for v in s]) + 3.0
    # 0cm をはさんで緑→赤。色の範囲は ±20cm で頭打ち
    traces = [go.Scatter3d(
        x=xy[:, 0], y=xy[:, 1], z=z, mode="lines+markers",
        line=dict(color=c, colorscale=[[0, NG_COLOR], [0.5, "#f1c40f"], [1, "#27ae60"]],
                  cmin=-20, cmax=20, width=8),
        marker=dict(size=3, color=c, colorscale=[[0, NG_COLOR], [0.5, "#f1c40f"], [1, "#27ae60"]],
                    cmin=-20, cmax=20, colorbar=dict(title="余裕(cm)", len=0.6)),
        text=[f"s={a:.0f}cm 余裕{b:+.1f}cm" for a, b in zip(s, c)],
        hoverinfo="text", showlegend=False)]
    full = np.column_stack([cl.xy, cl.z + 3.0])
    traces.append(go.Scatter3d(x=full[:, 0], y=full[:, 1], z=full[:, 2], mode="lines",
                               line=dict(color="#999999", width=2, dash="dot"),
                               hoverinfo="skip", showlegend=False))
    return traces


def _scene_traces(space, mesh, bottleneck, with_human, num_carriers, crop_r):
    """一番狭い所のまわりの面・家具・運搬者・当たった点のトレース。"""
    pos, quat = np.array(bottleneck["pose"][0]), np.array(bottleneck["pose"][1])
    ng = bottleneck["capacity_cm"] < 0
    color = NG_COLOR if ng else OK_COLOR
    traces = [_mesh_trace(_crop_mesh(mesh, pos, crop_r))]
    traces.append(V.box_mesh(*V.furniture_state(pos, quat), color, opacity=0.9))
    if with_human:
        cl = space.centerline
        for cp in L.carrier_poses_lstair(pos, quat, num_carriers, space, None,
                                         space.floor_fn):
            # 立つ床がない所(壁の外など)では、宙に浮かせず近くの中心線の床に
            # 立たせる。壁を突き抜けて見えるので「ここには立てない」と分かる
            x, y = cp["foot_xy"]
            if space.floor_fn(x, y, cp["center"][2]) is None:
                cp = dict(cp, floor_z=cl.z_at(cl.project(x, y)))
            traces.append(V.human_mesh(cp, NG_COLOR if ng else CARRIER_COLOR, opacity=0.85))
    ct = bottleneck["contact"]
    xyz = ct["xyz"]
    label = ("家具" if ct["part"] == "furniture" else "運搬者の体") + \
        f"がここで{abs(ct['clearance_cm']):.1f}cm" + ("めり込む" if ct["clearance_cm"] < 0 else "の余裕")
    traces.append(go.Scatter3d(x=[xyz[0]], y=[xyz[1]], z=[xyz[2]], mode="markers+text",
                               marker=dict(size=9, color=NG_COLOR if ng else "#27ae60",
                                           symbol="diamond", line=dict(color="black", width=2)),
                               text=[label], textposition="top center",
                               hoverinfo="text", showlegend=False))
    return traces


def render_html(space, mesh, result, out_path, num_carriers=2, crop_r=250.0, title=None,
                include_plotlyjs=True):
    """scan_plan.judge の結果(bottleneck / bottleneck_furniture_only と掃引の
    profile を含む dict)を、左=家具だけ・右=運搬者ありの3Dで書き出す。

    include_plotlyjs は plotly の write_html にそのまま渡す(既定は埋め込み。
    ローカルサーバーから配信するときは "/plotly.min.js" のようにURLを渡す)。"""
    cases = (("bottleneck_furniture_only", False, "家具だけ"),
             ("bottleneck", True, f"{num_carriers}人で運ぶ"))
    titles = []
    for key, _, name in cases:
        b = result[key]
        verdict = (f"一番狭い所で余裕{b['capacity_cm']:+.1f}cm" if b["capacity_cm"] >= 0
                   else f"一番狭い所であと{-b['capacity_cm']:.1f}cm足りない")
        titles.append(f"{name}: {verdict}")
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
                        subplot_titles=titles, horizontal_spacing=0.02)
    for col, (key, with_human, _) in enumerate(cases, start=1):
        b = result[key]
        sweep = [(s, c, None) for s, c in b["profile"]]
        for tr in _centerline_trace(space, sweep):
            fig.add_trace(tr, row=1, col=col)
        for tr in _scene_traces(space, mesh, b, with_human, num_carriers, crop_r):
            fig.add_trace(tr, row=1, col=col)
    scene = dict(aspectmode="data", camera=dict(eye=dict(x=-1.2, y=-1.4, z=1.0)))
    fig.update_layout(scene=scene, scene2=scene, height=700,
                      title=dict(text=title or "Odoriba: スキャンした空間での判定"
                                 " -- 赤=足りない所 / 印=当たった点(ドラッグで回転)", x=0.5),
                      margin=dict(l=0, r=0, t=80, b=0))
    fig.write_html(out_path, include_plotlyjs=include_plotlyjs)
    print(f"saved viewer to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", nargs="?", help="スキャンのOBJ/PLY(--route なら不要)")
    parser.add_argument("--route", action="store_true", help="検証用の合成経路で表示する")
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="自由空間の中の1点(既定は出発点の床から1m上)")
    parser.add_argument("--carriers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--furniture", type=float, nargs=3, metavar=("L", "W", "H"),
                        help=f"家具の寸法(cm、長辺・短辺・高さ)。既定は {L.FURN_L:.0f} "
                             f"{L.FURN_W:.0f} {L.FURN_H:.0f}")
    parser.add_argument("--rrt", action="store_true",
                        help="RRT探索も回す(遅い。既定は掃引だけ)")
    parser.add_argument("--out", default="scan_viewer.html")
    args = parser.parse_args()
    if args.furniture:
        L.FURN_L, L.FURN_W, L.FURN_H = sorted(args.furniture[:2], reverse=True) +             [args.furniture[2]]

    import scan_space as SS
    if args.route:
        import mesh_env as M
        import route_env as R
        mesh = M.env_to_mesh(*R.build_route())
        mesh.invert()   # スキャンと同じ「面が自由空間側を向く」向きにする
        start, goal = R.START_XYZ, R.GOAL_XYZ
        seed = (R.HALL_W / 2, 40.0, 100.0)
    else:
        if not args.mesh:
            parser.error("mesh を指定する(または --route)")
        import scan_demo as SD
        mesh, _ = SD.load_scan_zup(args.mesh)
        start, goal = args.start, args.goal
        if args.seed:
            seed = args.seed
        elif start:
            seed = (start[0], start[1], start[2] + 100.0)
        else:
            seed = SS.auto_seed(mesh)

    space = SS.ScanSpace(mesh, seed)
    if start is None or goal is None:
        # 出発点・目的地の指定がなければ、床の上を歩いて一番遠い2点を使う
        start, goal, walk = space.propose_endpoints()
        print(f"出発点・目的地を自動で提案: {np.round(start, 0).tolist()} -> "
              f"{np.round(goal, 0).tolist()} (歩いて{walk:.0f}cm)")
    space.build_centerline(start, goal)
    result = SP.judge(space, num_carriers=args.carriers, rrt=args.rrt)
    render_html(space, mesh, result, args.out, num_carriers=args.carriers)


if __name__ == "__main__":
    main()
