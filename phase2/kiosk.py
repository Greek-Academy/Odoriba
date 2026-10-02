"""
展示用の判定画面(デジタル学園祭向け)。

来場者がブラウザで階段と家具の寸法を入れると、「家具だけなら通るか」
「2人で運ぶと通るか・あと何cm足りないか」を返す。RRTは回さず、
stair_report --quick と同じ掃引(1回あたり十数秒)だけを使う。

    cd phase2
    python kiosk.py            # http://localhost:8000 を開く

標準ライブラリの http.server だけで動く(追加インストール不要)。
demo/ 以下のファイル(3Dビューア・説明図)も /demo/ で配信する。
"""

import html
import json
import threading
import time
import traceback
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import phase2_lstair as L
import phase2_lstair_3d as V

PORT = 8000
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
# 会場がオフラインでも3Dが出るよう、plotly同梱のJSを配信する
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"

# 選ぶだけで試せるプリセット(起動時に裏で計算しておき、押したら即答する)
STAIR_PRESETS = {
    "ふつうの家の階段(幅80cm)": dict(width=80, rise=18, tread=25, steps=10),
    "広めの階段(幅90cm)": dict(width=90, rise=18, tread=26, steps=8),
    "狭い階段(幅75cm)": dict(width=75, rise=20, tread=23, steps=10),
}
FURN_PRESETS = {
    "タンス 200×50×65": (200, 50, 65),
    "2人掛けソファ 160×85×80": (160, 85, 80),
    "本棚 180×40×30": (180, 40, 30),
    "冷蔵庫 180×65×65": (180, 65, 65),
    # 2人運搬でもふつうの階段(幅80)を余裕1cmで通る、ぎりぎりの対比用
    "小さめの収納棚 80×45×40": (80, 45, 40),
    "段ボール箱 60×42×40": (60, 42, 40),
}

# 計算は1件ずつ(phase2_lstairがモジュール変数で寸法を持つため並列不可)
_lock = threading.Lock()
_cache = {}


def _bottleneck(outer, obstacles, with_human):
    """掃引して(場所, 最小クリアランスcm, その位置での最良姿勢)を返す。
    姿勢が1つも取れなければ(None, -inf, None)。"""
    sweep = L.sweep_capacity(outer, obstacles, num_carriers=2 if with_human else None,
                             with_human=with_human)
    finite = [(s, c, pose) for s, c, pose in sweep if np.isfinite(c)]
    if not finite:
        return None, -np.inf, None
    s_min, c_min, pose = min(finite, key=lambda r: r[1])
    s_land0, s_land1 = L.FL1_Y1, L.S_CORNER + (L.FL2_X0 - L.CENTER)
    where = "踊り場" if s_land0 <= s_min <= s_land1 else "階段の途中"
    return where, c_min, pose


def _figure_json(outer, obstacles, pose_b, c_b, pose_h, c_h):
    """一番狭い場所での家具(左)と、2人で運ぶ様子(右)の3D図をJSONで返す。
    足りない側は家具と運搬者を赤くする。"""
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
                        subplot_titles=("家具だけ(一番狭い場所)",
                                        "2人で持って運ぶ(一番狭い場所)"),
                        horizontal_spacing=0.02)
    for col in (1, 2):
        for center, rot, half in obstacles:
            fig.add_trace(V.box_mesh(center, rot, half, "#cfc8ba"), row=1, col=col)
        fig.add_trace(V.outer_wireframe(outer), row=1, col=col)
    for col, pose, c in ((1, pose_b, c_b), (2, pose_h, c_h)):
        if pose is None:
            continue
        pos, quat = np.array(pose[0]), np.array(pose[1])
        ng = c < 0
        fig.add_trace(V.box_mesh(*V.furniture_state(pos, quat),
                                 "#c0392b" if ng else "#2f4a5e"), row=1, col=col)
        if col == 2:
            for cc, r, hh in V.carrier_states(pos, quat, outer, obstacles, 2):
                fig.add_trace(V.cylinder_mesh(cc, r, hh, "#d9604f" if ng else "#b08d57",
                                              opacity=0.85), row=1, col=col)
    scene = dict(aspectmode="data", camera=dict(eye=dict(x=-1.3, y=-1.5, z=0.9)))
    fig.update_layout(scene=scene, scene2=scene, height=560,
                      paper_bgcolor="#fbfaf6", font=dict(color="#3d3b37"),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig.to_json()


def judge(width, rise, tread, steps, fl, fw, fh):
    """寸法を受け取り判定結果のdictを返す(同じ入力はキャッシュ)。"""
    key = (width, rise, tread, steps, fl, fw, fh)
    with _lock:
        if key in _cache:
            return _cache[key]
        t0 = time.time()
        L.configure(width=width, rise=rise, tread=tread, n_steps1=steps, n_steps2=steps)
        # 長辺・短辺・高さの順に揃える(掃引は長辺=FURN_Lを前提にしている)
        a, b = sorted([fl, fw], reverse=True)
        L.FURN_L, L.FURN_W, L.FURN_H = float(a), float(b), float(fh)
        outer, obstacles = L.build_lstairs()
        where_b, c_b, pose_b = _bottleneck(outer, obstacles, with_human=False)
        where_h, c_h, pose_h = _bottleneck(outer, obstacles, with_human=True)
        fig = _figure_json(outer, obstacles, pose_b, c_b, pose_h, c_h)
        res = dict(where_b=where_b, c_b=float(c_b), where_h=where_h, c_h=float(c_h),
                   fig=fig, sec=time.time() - t0)
        _cache[key] = res
        return res


def _warm_presets():
    """起動直後に、プリセットの組み合わせを裏で計算しておく。"""
    for st in STAIR_PRESETS.values():
        for f in FURN_PRESETS.values():
            try:
                judge(st["width"], st["rise"], st["tread"], st["steps"], *f)
            except Exception:
                traceback.print_exc()
    print("[kiosk] プリセットの事前計算が完了しました")


def _verdict_html(c, where, label):
    if not np.isfinite(c):
        return (f'<div class="card ng"><h3>{label}</h3>'
                f'<p class="big">✕ 通らない</p><p>どの向きでも入りません</p></div>')
    if c >= 0:
        return (f'<div class="card ok"><h3>{label}</h3><p class="big">○ 通る</p>'
                f'<p>一番狭いのは{where}、余裕 <b>{c:.0f}cm</b></p></div>')
    return (f'<div class="card ng"><h3>{label}</h3><p class="big">✕ 通らない</p>'
            f'<p>{where}で あと <b>{-c:.0f}cm</b> 足りない</p></div>')


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>Odoriba — その家具、運び込めますか？</title>
<style>
body{{font-family:"Yu Gothic UI","Meiryo",sans-serif;max-width:1000px;margin:28px auto;padding:0 20px;background:#f4f1ea;color:#2b2a28;letter-spacing:.02em}}
h1,h2,h3{{font-family:"Yu Mincho","YuMincho","Hiragino Mincho ProN",serif;font-weight:600}}
h1{{font-size:2.4em;margin:.2em 0;color:#1f2a24;letter-spacing:.08em}}
h2{{border-left:3px solid #8a7350;padding-left:10px;color:#1f2a24;margin-top:1.6em}}
.lead{{font-size:1.15em;line-height:1.8;color:#3d3b37}}
form{{background:#fbfaf6;padding:20px 22px;border:1px solid #d9d2c3;border-radius:4px}}
fieldset{{border:none;margin:0 0 10px;padding:0}} legend{{font-weight:bold;font-size:1.05em;color:#1f2a24}}
label{{display:inline-block;margin:4px 14px 4px 0;font-size:1.02em;color:#3d3b37}}
input[type=number]{{width:5em;font-size:1.05em;padding:3px 6px;border:1px solid #bfb6a3;border-radius:3px;background:#fff}}
.presets button{{font-size:.98em;margin:3px;padding:7px 12px;border-radius:3px;border:1px solid #8a7350;background:transparent;color:#4a3f2e;cursor:pointer;transition:.15s}}
.presets button:hover{{background:#8a7350;color:#fbfaf6}}
.go{{font-size:1.25em;padding:10px 44px;background:#1f3a2e;color:#f4f1ea;border:none;border-radius:3px;cursor:pointer;letter-spacing:.2em}}
.go:hover{{background:#2d5242}}
.cards{{display:flex;gap:18px;margin-top:16px}} .card{{flex:1;padding:16px 20px;border-radius:4px;background:#fbfaf6}}
.card h3{{margin:.2em 0;color:#3d3b37;font-size:1.05em}}
.ok{{border:1px solid #1f3a2e;border-top:5px solid #1f3a2e}} .ok .big{{color:#1f3a2e}}
.ng{{border:1px solid #7a2e2e;border-top:5px solid #7a2e2e}} .ng .big{{color:#7a2e2e}}
.big{{font-family:"Yu Mincho","YuMincho",serif;font-size:2.2em;font-weight:bold;margin:.15em 0}}
.msg{{font-size:1.08em;line-height:1.8;background:#ece6d8;border-left:3px solid #8a7350;padding:12px 16px;margin-top:14px}}
#wait{{display:none;font-size:1.15em;color:#8a7350;margin-top:10px}}
.links a{{display:inline-block;margin:4px 16px 4px 0;font-size:1.02em;color:#1f3a2e;text-decoration:none;border-bottom:1px solid #8a7350}}
.links a:hover{{color:#8a7350}}
.note{{color:#7a7468;font-size:.9em;line-height:1.7}}
#viz{{background:#fbfaf6;border:1px solid #d9d2c3;border-radius:4px}}
</style></head><body>
<h1>Odoriba（踊り場）</h1>
<p class="lead">その家具、<b>部屋に置けても、そこまで運べますか？</b><br>
家具だけなら通るのに、<b>人が持つと</b>踊り場で回せない——を計算で見つけます。</p>

<form method="get" action="/" onsubmit="document.getElementById('wait').style.display='block'">
<fieldset class="presets"><legend>① 階段をえらぶ（L字に曲がる階段）</legend>
{stair_buttons}</fieldset>
<label>階段の幅 <input type="number" name="width" value="{width}" min="50" max="150"> cm</label>
<label>1段の高さ <input type="number" name="rise" value="{rise}" min="12" max="25"> cm</label>
<label>1段の奥行き <input type="number" name="tread" value="{tread}" min="18" max="35"> cm</label>
<label>段数(踊り場まで) <input type="number" name="steps" value="{steps}" min="3" max="15"> 段</label>
<fieldset class="presets" style="margin-top:12px"><legend>② 家具をえらぶ（押すとすぐ判定します）</legend>
{furn_buttons}</fieldset>
<label>長さ <input type="number" name="fl" value="{fl}" min="20" max="300"> cm</label>
<label>幅 <input type="number" name="fw" value="{fw}" min="20" max="200"> cm</label>
<label>高さ <input type="number" name="fh" value="{fh}" min="20" max="200"> cm</label>
<p><button class="go" type="submit" name="go" value="1">判定する</button></p>
<p id="wait">計算中です… 新しい寸法だと20〜40秒ほどかかります</p>
</form>
{result}
<script src="/plotly.min.js"></script>
<script>
const FIG = {fig};
if (FIG) Plotly.newPlot('viz', FIG.data, FIG.layout, {{responsive: true}});
</script>
<h2>くわしく見る</h2>
<div class="links">{links}</div>
<p class="note">運ぶ人は2人、体を円柱(半径20cm・身長170cm)で近似。家具を傾けられるのは55度まで、
持つ高さは10〜190cmと仮定しています。結果は研究中の試作品によるものです。</p>
<script>
function setv(o){{for(const k in o)document.querySelector('[name='+k+']').value=o[k];}}
// 家具ボタンは選んだらそのまま計算する
function pick(o){{setv(o);document.getElementById('wait').style.display='block';
  const f=document.querySelector('form');const g=document.createElement('input');
  g.type='hidden';g.name='go';g.value='1';f.appendChild(g);f.submit();}}
</script>
</body></html>"""

LINKS = [
    ("ustair_explain.png", "説明図：折り返し階段（1枚でわかる）"),
    ("case_single.html", "3D：家具だけで運ぶ"),
    ("case_carriers.html", "3D：2人で運ぶと詰まる"),
    ("ustair_3d.html", "3D：折り返し階段"),
    ("phase2_lstair_3d.html", "3D：L字階段"),
    ("box_pass.gif", "アニメ：段ボール箱"),
]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DEMO_DIR), **kw)

    def log_message(self, fmt, *args):
        pass  # 展示中にコンソールが流れないよう黙らせる

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/plotly.min.js":
            data = PLOTLY_JS.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(data)
            return
        if u.path.startswith("/demo/"):
            self.path = u.path[len("/demo"):]
            return super().do_GET()
        if u.path != "/":
            self.send_error(404)
            return
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        st = dict(STAIR_PRESETS["ふつうの家の階段(幅80cm)"])
        fl, fw, fh = FURN_PRESETS["タンス 200×50×65"]
        result = ""
        fig = "null"
        try:
            st = {k: float(q.get(k, st[k])) for k in st}
            st["steps"] = int(st["steps"])
            fl, fw, fh = (float(q.get(k, d)) for k, d in (("fl", fl), ("fw", fw), ("fh", fh)))
            if "go" in q:
                st["width"] = min(max(st["width"], 50), 150)
                st["rise"] = min(max(st["rise"], 12), 25)
                st["tread"] = min(max(st["tread"], 18), 35)
                st["steps"] = min(max(st["steps"], 3), 15)
                fl, fw, fh = (min(max(v, 20), 300) for v in (fl, fw, fh))
                r = judge(st["width"], st["rise"], st["tread"], st["steps"], fl, fw, fh)
                result = '<h2>結果</h2><div class="cards">' + \
                    _verdict_html(r["c_b"], r["where_b"], "家具だけなら") + \
                    _verdict_html(r["c_h"], r["where_h"], "2人で持って運ぶと") + "</div>"
                result += ('<p class="note">下の3Dは、一番狭い場所で家具を一番うまく向けた姿勢です。'
                           'マウスでドラッグすると回せます。</p>'
                           '<p class="note"><span style="color:#c0392b">■</span> 赤＝足りない　'
                           '<span style="color:#2f4a5e">■</span> 紺＝通る　'
                           '<span style="color:#b08d57">●</span> 金色の柱＝運ぶ人</p><div id="viz"></div>')
                fig = r["fig"].replace("</", "<\\/")
                if r["c_b"] >= 0 > r["c_h"]:
                    result += ('<p class="msg">家具だけなら通るのに、人が持つと通らない！<br>'
                               '運ぶ人の体も通路をふさぐので、踊り場で家具を回すスペースが足りなくなります。'
                               '既存の「家具が置けるか」アプリでは見落とされるケースです。</p>')
        except Exception as e:
            traceback.print_exc()
            result = f'<p class="msg">計算できませんでした（{html.escape(str(e))}）。寸法を変えて試してください。</p>'

        def btn(name, js):
            return f'<button type="button" onclick=\'setv({js})\'>{html.escape(name)}</button>'
        stair_buttons = "".join(btn(n, json.dumps(v)) for n, v in STAIR_PRESETS.items())
        furn_buttons = "".join(btn(n, json.dumps(dict(fl=f[0], fw=f[1], fh=f[2])))
                               for n, f in FURN_PRESETS.items())
        links = "".join(f'<a href="/demo/{f}" target="_blank">{t}</a>'
                        for f, t in LINKS if (DEMO_DIR / f).exists())
        body = PAGE.format(stair_buttons=stair_buttons, furn_buttons=furn_buttons,
                           result=result, fig=fig, links=links, fl=f"{fl:g}", fw=f"{fw:g}",
                           fh=f"{fh:g}", **{k: f"{v:g}" for k, v in st.items()})
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    threading.Thread(target=_warm_presets, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[kiosk] http://localhost:{PORT} で起動しました(Ctrl+Cで終了)")
    webbrowser.open(f"http://localhost:{PORT}")
    srv.serve_forever()
