"""
L字階段の3Dビューア: results/*.json の保存済み経路を、ブラウザで
回せる3D表示(HTML)として書き出す。

    python phase2_lstair_3d.py                        # results/lstair_result.json から
    python phase2_lstair_3d.py --json <path> --out <html>

探索は再実行しない(JSONの経路をそのまま再生する)。出力HTMLは
plotly.jsをCDNから読むため、ファイル1つ+ネット接続だけで誰でも開ける
(環境構築不要、というのがこのビューアの目的)。

左=家具単体(PASS) / 右=運搬者あり(BLOCKED、探索が到達できた所まで)を
並置し、再生ボタン/スライダーで両方が同時に動く。右は詰まった時点で
家具が赤くなる。
"""

import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import geometry3d as g3
import phase2_lstair as L

N_FRAMES = 80          # 再生フレーム数(経路はこの数に間引く)
CYL_SEGMENTS = 14      # 運搬者円柱の側面の分割数


# ---- メッシュ生成 ----

# g3.obb_cornersの頂点順(sx, sy, szの辞書順)に対応する12三角形。
_BOX_TRI = np.array([
    (0, 1, 3), (0, 3, 2),   # -x面
    (4, 6, 7), (4, 7, 5),   # +x面
    (0, 4, 5), (0, 5, 1),   # -y面
    (2, 3, 7), (2, 7, 6),   # +y面
    (0, 2, 6), (0, 6, 4),   # -z面
    (1, 5, 7), (1, 7, 3),   # +z面
])


def box_mesh(center, rotmat, half, color, opacity=1.0, name=None):
    v = g3.obb_corners(np.asarray(center), rotmat, np.asarray(half))
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                     i=_BOX_TRI[:, 0], j=_BOX_TRI[:, 1], k=_BOX_TRI[:, 2],
                     color=color, opacity=opacity, flatshading=True,
                     name=name, showlegend=False, hoverinfo="skip")


def _cylinder_verts_tris(center, radius, half_h, n=CYL_SEGMENTS):
    """縦向き円柱の頂点と三角形(側面+上下の蓋)。"""
    cx, cy, cz = center
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1)
    bot = np.column_stack([ring, np.full(n, cz - half_h)])
    top = np.column_stack([ring, np.full(n, cz + half_h)])
    verts = np.vstack([bot, top, [[cx, cy, cz - half_h]], [[cx, cy, cz + half_h]]])
    c_bot, c_top = 2 * n, 2 * n + 1
    tris = []
    for a in range(n):
        b = (a + 1) % n
        tris.append((a, b, n + a))            # 側面
        tris.append((b, n + b, n + a))
        tris.append((c_bot, b, a))            # 下蓋
        tris.append((c_top, n + a, n + b))    # 上蓋
    return verts, np.array(tris)


def cylinder_mesh(center, radius, half_h, color, opacity=1.0):
    v, t = _cylinder_verts_tris(center, radius, half_h)
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                     i=t[:, 0], j=t[:, 1], k=t[:, 2],
                     color=color, opacity=opacity, flatshading=True,
                     showlegend=False, hoverinfo="skip")


def outer_wireframe(outer):
    """外枠(自由空間の輪郭)の直方体をワイヤーフレームで描く。
    面で塗ると中の家具が見えなくなるため線だけにする。"""
    xs, ys, zs = [], [], []
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for center, rot, half in outer:
        v = g3.obb_corners(center, rot, half)
        for a, b in edges:
            xs += [v[a, 0], v[b, 0], None]
            ys += [v[a, 1], v[b, 1], None]
            zs += [v[a, 2], v[b, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color="#b0b0b0", width=2),
                        showlegend=False, hoverinfo="skip")


# ---- 経路の再生データ ----

def parse_path(raw):
    return [(np.array(s[:3]), np.array(s[3:])) for s in raw]


def resample(path, n):
    """経路をnフレームに間引く(最後の状態は必ず含める)。"""
    idx = np.round(np.linspace(0, len(path) - 1, n)).astype(int)
    return [path[i] for i in idx]


def furniture_state(pos, quat):
    half = np.array([L.FURN_L / 2, L.FURN_W / 2, L.FURN_H / 2])
    return np.asarray(pos), g3.rotmat_from_quat(quat), half


def carrier_states(pos, quat, outer, obstacles, num_carriers):
    """このフレームでの運搬者円柱(中心, 半径, 半高)のリスト。"""
    out = []
    for c in L.best_human_positions_lstair(pos, quat, num_carriers, outer, obstacles):
        out.append((np.asarray(c), L.p.HUMAN_R, L._CARRIER_HALF_H))
    return out


def render_html(result, out_path, use_cdn=False):
    outer, obstacles = L.build_lstairs()
    meta = result["meta"]
    num_carriers = meta["carrier"]["num_carriers"]

    box = result["box_only"]
    car = result["with_carriers"]
    path_box = resample(parse_path(box["path"] if box["found"]
                                   else box["best_effort_path"]), N_FRAMES)
    path_car = resample(parse_path(car["path"] if car["found"]
                                   else car["best_effort_path"]), N_FRAMES)
    # 運搬者の立ち位置はフレームごとにオラクルで求めて先に計算しておく
    carriers_car = [carrier_states(pos, q, outer, obstacles, num_carriers)
                    for pos, q in path_car]

    titles = (
        f"furniture only: {'PASS' if box['found'] else 'BLOCKED'}",
        f"+ {num_carriers} carriers: "
        f"{'PASS' if car['found'] else 'BLOCKED (stops where search got stuck)'}",
    )
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
                        subplot_titles=titles, horizontal_spacing=0.02)

    # ---- 静的な環境(両シーンに同じものを描く) ----
    for col, scene in ((1, "scene"), (2, "scene2")):
        for center, rot, half in obstacles:
            fig.add_trace(box_mesh(center, rot, half, "#c9c9c9", opacity=1.0),
                          row=1, col=col)
        fig.add_trace(outer_wireframe(outer), row=1, col=col)

    # ---- 動的トレース(フレームで差し替える分)。追加順を覚えておく ----
    dyn_start = len(fig.data)
    p0, r0, h0 = furniture_state(*path_box[0])
    fig.add_trace(box_mesh(p0, r0, h0, "#2c6fbb"), row=1, col=1)      # 左の家具
    p0, r0, h0 = furniture_state(*path_car[0])
    fig.add_trace(box_mesh(p0, r0, h0, "#2c6fbb"), row=1, col=2)      # 右の家具
    for c, r, hh in carriers_car[0]:                                   # 右の運搬者
        fig.add_trace(cylinder_mesh(c, r, hh, "#e07b39", opacity=0.85), row=1, col=2)
    dyn_idx = list(range(dyn_start, len(fig.data)))

    # ---- フレーム ----
    frames = []
    for k in range(N_FRAMES):
        p, r, h = furniture_state(*path_box[k])
        data = [box_mesh(p, r, h, "#2c6fbb")]
        p, r, h = furniture_state(*path_car[k])
        # 右は最終フレーム(=詰まった状態)で赤にする
        stuck = (not car["found"]) and k == N_FRAMES - 1
        data.append(box_mesh(p, r, h, "#c0392b" if stuck else "#2c6fbb"))
        for c, rad, hh in carriers_car[k]:
            data.append(cylinder_mesh(c, rad, hh,
                                      "#c0392b" if stuck else "#e07b39", opacity=0.85))
        frames.append(go.Frame(data=data, traces=dyn_idx, name=str(k)))
    fig.frames = frames

    # ---- 再生UI ----
    steps = [dict(method="animate", label="",
                  args=[[str(k)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for k in range(N_FRAMES)]
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=True),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)])])],
        sliders=[dict(steps=steps, x=0.12, len=0.85, y=0.04,
                      currentvalue=dict(visible=False))],
        title=dict(text=(
            "Odoriba Phase 2: L-shaped staircase 3D viewer "
            f"(stair width {meta['stair']['width']:.0f}cm, "
            f"furniture {meta['furniture']['L']:.0f}x{meta['furniture']['W']:.0f}"
            f"x{meta['furniture']['H']:.0f}cm) -- drag to rotate"
            # 結論の数字(PNG/GIFと同じ出典=結果JSON)をタイトル2行目に。
            # 位置指定のannotationはスライダーと重なりやすいのでtitleに載せる
            + ("<br><sup>" + "  |  ".join(L.result_summary_lines(result)) + "</sup>"
               if L.result_summary_lines(result) else "")),
            x=0.5),
        margin=dict(l=0, r=0, t=70, b=0),
    )
    same_scene = dict(aspectmode="data",
                      camera=dict(eye=dict(x=-1.3, y=-1.5, z=0.9)))
    fig.update_layout(scene=same_scene, scene2=same_scene)

    # 既定はplotly.jsをHTMLに埋め込む(約4MB増えるが、合宿など
    # ネットの無い場所でもファイル1つで開ける)。--cdnで軽量版。
    fig.write_html(out_path, include_plotlyjs=("cdn" if use_cdn else True),
                   auto_play=False)
    print(f"saved 3D viewer to {out_path}"
          f" ({'CDN' if use_cdn else 'offline, plotly.js embedded'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=os.path.join("results", "lstair_result.json"))
    parser.add_argument("--out", default="phase2_lstair_3d.html")
    parser.add_argument("--cdn", action="store_true",
                        help="plotly.jsを埋め込まずCDNから読む(ファイルが小さくなるが"
                             "ネット接続が必要)")
    args = parser.parse_args()
    with open(args.json) as f:
        result = json.load(f)
    render_html(result, args.out, use_cdn=args.cdn)
