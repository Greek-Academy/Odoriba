"""
スキャンした家具(段ボール等)を「運ぶ剛体」として判定に使う。

これまで家具は直方体(FURN_L x W x H)の近似だったが、ここではスキャンした
メッシュの表面点をそのまま家具の形として使い、L字階段(phase2_lstair)を
通れるか(通る/通らない・最小クリアランス)を判定する。段ボールのような
閉じた物体はスキャンが水密化しやすく、家具スキャンの実証に向く。

パイプライン:
  scan_ingestで単位をcmに正規化 -> OBB(有向境界箱)で「長辺=ローカルx」に
  向きを揃えて中心を原点へ -> 表面点をサンプル -> その点群を家具として
  L字階段のクリアランス評価・RRT探索にかける。

いまの範囲: 家具単体(運搬者なし)の通過判定。運搬者ありは把持点の定義を
スキャン家具に合わせる必要があり、次段階。

実行:
    python scan_furniture.py <furniture.obj>              # L字階段を通れるか
    python scan_furniture.py <furniture.obj> --max-iter 2000
    python scan_furniture.py --selftest                  # 合成の箱で自己テスト
"""

import argparse

import numpy as np
import trimesh

import phase2_demo as p
import phase2_lstair as L
import scan_ingest as si
import scan_demo as sd


def isolate_object_points(path, assume_units=None, min_range=20.0, xy_bin=5.0):
    """床や周囲を一緒に撮ってしまったスキャンから、対象物(段ボール等)の
    点だけを切り出す。

    考え方: 床は薄い板なので、そのXYセルの高さの幅(zの最大-最小)は小さい。
    箱は壁で床から天面まで厚みがあるので、そのセルの高さの幅は大きい。
    そこで「高さの幅が min_range cm 以上のセル」を対象物の footprint とし、
    その最大連結塊(+余白)の点を切り出す。上下の向きに依存しないので、
    軸の反転や床の広さに強い。
    戻り値: 対象物の点群(N,3, cm)
    """
    from scipy import ndimage
    mesh, _ = sd.load_scan_zup(path, assume_units)
    v = np.asarray(mesh.vertices)
    bx = np.arange(v[:, 0].min(), v[:, 0].max() + xy_bin, xy_bin)
    by = np.arange(v[:, 1].min(), v[:, 1].max() + xy_bin, xy_bin)
    ix = np.clip(np.digitize(v[:, 0], bx) - 1, 0, len(bx) - 2)
    iy = np.clip(np.digitize(v[:, 1], by) - 1, 0, len(by) - 2)
    nx, ny = len(bx) - 1, len(by) - 1
    zmax = np.full((nx, ny), -1e9)
    zmin = np.full((nx, ny), 1e9)
    np.maximum.at(zmax, (ix, iy), v[:, 2])
    np.minimum.at(zmin, (ix, iy), v[:, 2])
    zrange = np.where(zmax > -1e8, zmax - zmin, 0.0)

    mask = zrange > min_range
    lbl, n = ndimage.label(mask)
    if n == 0:
        return v  # 厚みのある塊が無い=切り出せない。そのまま返す
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    xi, yi = np.where(lbl == sizes.argmax())
    x0, x1 = bx[xi.min()], bx[xi.max() + 1]
    y0, y1 = by[yi.min()], by[yi.max() + 1]

    m = 3.0
    sel = ((v[:, 0] >= x0 - m) & (v[:, 0] <= x1 + m) &
           (v[:, 1] >= y0 - m) & (v[:, 1] <= y1 + m))
    return v[sel]


def load_furniture(path=None, mesh=None, points=None, assume_units=None, n_points=250):
    """家具スキャンを読み、OBB(有向境界箱)で姿勢を正規化して表面点
    (ローカル・中心原点・長辺がx)を返す。

    入力はpath / mesh / points のいずれか。pointsは切り出し済みの点群。
    戻り値: dict(local=表面点(N,3), dims=(L,W,H降順))
    """
    if points is None:
        if mesh is None:
            mesh = si.load_mesh(path)
            scale, _ = si.guess_units_scale(mesh, assume_units)
            mesh.apply_scale(scale)
        try:
            pts = np.asarray(mesh.sample(n_points * 4))
        except Exception:
            pts = np.asarray(mesh.vertices)
    else:
        pts = np.asarray(points)

    # 点群の有向境界箱で、主軸を座標軸に揃えて中心を原点へ
    T, ext = trimesh.bounds.oriented_bounds(pts)
    local = trimesh.transform_points(pts, T)
    order = np.argsort(ext)[::-1]  # 長い順 -> x=長辺, y=中間, z=短辺
    local = local[:, order]
    dims = np.asarray(ext)[order]

    if len(local) > n_points:
        idx = np.random.default_rng(0).choice(len(local), n_points, replace=False)
        local = local[idx]
    return {"local": local, "dims": dims}


def furniture_world_points(local, pos, quat):
    """ローカル表面点を姿勢(pos,quat)でワールド座標へ。
    phase2_lstair.furniture_pointsのスキャン家具版。"""
    R = p.g3.rotmat_from_quat(quat)
    return (np.asarray(local) @ R.T) + np.asarray(pos)


def _start_goal(h):
    """家具高さhに合わせたSTART/GOAL(catalog.pyと同じ組み立て)。"""
    start = (np.array([L.CENTER, 170.0, h / 2]), L._pose(90.0, 0.0).as_quat())
    goal = (np.array([L.FL2_X1 + 170.0, L.LAND_YC, L.TOP_Z + h / 2]),
            L._pose(0.0, 0.0).as_quat())
    return start, goal


def judge_in_lstair(furn, max_iter=3000, seeds=(0, 1)):
    """スキャン家具がL字階段を通れるか(家具単体)。

    家具の外形寸法(dims)をL.FURN_*に入れてサンプラー・START/GOALを
    整えつつ、クリアランス判定だけはスキャン点群で行う(validatorを差し替え)。
    戻り値: dict(found, min_clearance, dims)
    """
    local, dims = furn["local"], furn["dims"]
    orig = (L.FURN_L, L.FURN_W, L.FURN_H)
    L.FURN_L, L.FURN_W, L.FURN_H = map(float, dims)
    try:
        outer, obstacles = L.build_lstairs()
        start, goal = _start_goal(dims[2])

        def validator(pos, quat, with_human, outer, obstacles, num_carriers=None):
            # 家具単体のみ(with_humanは無視)。スキャン点群で最小クリアランス
            return p.shape_clearance_3d(
                furniture_world_points(local, pos, quat), outer, obstacles) >= 0

        found = False
        min_clear = None
        found_path = None
        for sd in seeds:
            path = p.rrt_connect(
                start, goal, with_human=False, outer=outer, obstacles=obstacles,
                max_iter=max_iter, seed=sd, w_rot=L.W_ROT,
                sampler=L.make_sampler(with_human=False), validator=validator)
            if path is not None:
                found = True
                found_path = path
                min_clear = min(
                    p.shape_clearance_3d(furniture_world_points(local, ps, q),
                                         outer, obstacles)
                    for ps, q in path)
                break
        return {"found": found, "min_clearance": min_clear, "dims": dims,
                "path": found_path}
    finally:
        L.FURN_L, L.FURN_W, L.FURN_H = orig


def render_gif(furn, path, out_path, fps=12, step_cm=10.0):
    """スキャン家具が経路に沿ってL字階段を通り抜けるGIF(運搬者なし=浮いて通る)。

    レイアウトはphase2_lstairのGIFと同じ2パネル(上面図/展開側面図)。
    家具はスキャンの点群をそのまま姿勢変換して散布図で描く。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import phase2_lstair_viz as viz
    import phase2_lstair_gif as gif

    local = furn["local"]
    outer, obstacles = L.build_lstairs()
    dense = gif.interpolate_path(path, L.W_ROT, step_cm=step_cm)

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(15, 6.5))

    def frame(k):
        ax_t.clear()
        ax_s.clear()
        viz.draw_env_top(ax_t, "top view")
        viz.draw_env_side(ax_s, "unfolded side view")
        pos, quat = dense[k]
        wp = furniture_world_points(local, pos, quat)
        ax_t.scatter(wp[:, 0], wp[:, 1], s=4, c="tab:blue", zorder=3)
        s = np.array([L.skeleton_s(x, y) for x, y, _ in wp])
        ax_s.scatter(s, wp[:, 2], s=4, c="tab:blue", zorder=3)
        d = furn["dims"]
        fig.suptitle(f"Scanned cardboard ({d[0]:.0f}x{d[1]:.0f}x{d[2]:.0f}cm) "
                     f"carried up the L-staircase (furniture floating, no carriers)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    anim = animation.FuncAnimation(fig, frame, frames=len(dense), interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved gif to {out_path} ({len(dense)} frames)")


def render_3d(furn, path, out_path, n_frames=60):
    """スキャン家具がL字階段を通り抜ける様子を、ブラウザで回せる3D(HTML)にする。

    phase2_lstair_3dの部品(階段の直方体・ワイヤーフレーム・再生UI)を
    再利用し、家具はスキャン寸法の箱として経路に沿って動かす。見た目重視の
    デモ用。plotly.jsを埋め込むのでネットなしでファイル1つで開ける。
    """
    import plotly.graph_objects as go
    import phase2_lstair_3d as v3
    import phase2_lstair_gif as gif

    outer, obstacles = L.build_lstairs()
    half = np.asarray(furn["dims"], dtype=float) / 2.0
    dense = gif.interpolate_path(path, L.W_ROT, step_cm=10.0)
    idx = np.linspace(0, len(dense) - 1, min(n_frames, len(dense))).round().astype(int)
    dense = [dense[i] for i in idx]

    def furn_mesh(pose, color):
        pos, quat = pose
        return v3.box_mesh(np.asarray(pos), p.g3.rotmat_from_quat(quat), half,
                           color, opacity=0.92)

    fig = go.Figure()
    for c, r, h in obstacles:  # 階段(灰)
        fig.add_trace(v3.box_mesh(c, r, h, "#c9c9c9", opacity=1.0))
    fig.add_trace(v3.outer_wireframe(outer))  # 階段室の輪郭
    fig.add_trace(furn_mesh(dense[0], "#2c6fbb"))  # 家具(初期)
    dyn = [len(fig.data) - 1]

    fig.frames = [go.Frame(data=[furn_mesh(pose, "#2c6fbb")], traces=dyn, name=str(k))
                  for k, pose in enumerate(dense)]
    steps = [dict(method="animate", label="",
                  args=[[str(k)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for k in range(len(dense))]
    d = furn["dims"]
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=True),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)])])],
        sliders=[dict(steps=steps, x=0.12, len=0.85, y=0.04,
                      currentvalue=dict(visible=False))],
        title=dict(text=f"Scanned cardboard ({d[0]:.0f}x{d[1]:.0f}x{d[2]:.0f}cm) "
                        "through the L-staircase -- drag to rotate", x=0.5),
        scene=dict(aspectmode="data", camera=dict(eye=dict(x=-1.4, y=-1.5, z=0.9))),
        margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(out_path, include_plotlyjs=True, auto_play=False)
    print(f"saved 3D viewer to {out_path}")


def _selftest():
    """合成の箱(段ボール想定)で、寸法の取り出しと通過判定が動くか。"""
    print("=== 自己テスト: 合成の箱を家具スキャンに見立てる ===")
    # 大箱は断面80x80(対角113cm>階段幅90cm)で、立てても回せず通らない寸法
    for name, ext in [("小さい箱 40x30x30", [40, 30, 30]),
                      ("大きい箱 250x80x80", [250, 80, 80])]:
        box = trimesh.creation.box(extents=ext)
        furn = load_furniture(mesh=box)
        print(f"{name}: 取り出したdims(cm)= {furn['dims'].round(0)}")
        res = judge_in_lstair(furn, max_iter=1500)
        verdict = "PASS" if res["found"] else "BLOCKED"
        mc = "" if res["min_clearance"] is None else f" (min clearance {res['min_clearance']:.1f}cm)"
        print(f"  L字階段: {verdict}{mc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("furniture", nargs="?", help="家具の OBJ/PLY")
    parser.add_argument("--assume-units", choices=("m", "cm", "mm"))
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--raw", action="store_true",
                        help="床除去(切り出し)をせず、スキャン全体を家具として使う")
    parser.add_argument("--gif", metavar="OUT",
                        help="通過経路を家具が浮いて通り抜けるGIFを書き出す")
    parser.add_argument("--html", metavar="OUT",
                        help="ブラウザで回せる3D(HTML)を書き出す(見た目重視)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    elif args.furniture:
        if args.raw:
            furn = load_furniture(args.furniture, assume_units=args.assume_units)
        else:
            pts = isolate_object_points(args.furniture, assume_units=args.assume_units)
            print(f"床・周囲を除去して対象物を切り出し: {len(pts)}点")
            furn = load_furniture(points=pts)
        print(f"家具スキャン: dims(cm)= {furn['dims'].round(0)} (長辺x中間x短辺)")
        res = judge_in_lstair(furn, max_iter=args.max_iter)
        verdict = "PASS(通る)" if res["found"] else "BLOCKED(この試行では見つからず)"
        mc = "" if res["min_clearance"] is None else f" / 最小クリアランス {res['min_clearance']:.1f}cm"
        print(f"L字階段(幅{L.STAIR_WIDTH:.0f}cm・踊り場{L.STAIR_WIDTH:.0f}角): {verdict}{mc}")
        if args.gif or args.html:
            if res["path"] is None:
                print("経路が見つからなかったので可視化は作れません(--max-iterを増やして再試行)")
            else:
                if args.gif:
                    render_gif(furn, res["path"], args.gif)
                if args.html:
                    render_3d(furn, res["path"], args.html)
    else:
        parser.error("furniture を指定するか --selftest を使ってください")
