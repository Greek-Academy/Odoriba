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

import phase2_lstair as L

PORT = 8000
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"

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
    "段ボール箱 60×42×40": (60, 42, 40),
}

# 計算は1件ずつ(phase2_lstairがモジュール変数で寸法を持つため並列不可)
_lock = threading.Lock()
_cache = {}


def _bottleneck(outer, obstacles, with_human):
    """掃引して(場所, 最小クリアランスcm)を返す。姿勢が1つも取れなければ(None, -inf)。"""
    sweep = L.sweep_capacity(outer, obstacles, num_carriers=2 if with_human else None,
                             with_human=with_human)
    finite = [(s, c) for s, c, _ in sweep if np.isfinite(c)]
    if not finite:
        return None, -np.inf
    s_min, c_min = min(finite, key=lambda r: r[1])
    s_land0, s_land1 = L.FL1_Y1, L.S_CORNER + (L.FL2_X0 - L.CENTER)
    where = "踊り場" if s_land0 <= s_min <= s_land1 else "階段の途中"
    return where, c_min


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
        where_b, c_b = _bottleneck(outer, obstacles, with_human=False)
        where_h, c_h = _bottleneck(outer, obstacles, with_human=True)
        res = dict(where_b=where_b, c_b=float(c_b), where_h=where_h, c_h=float(c_h),
                   sec=time.time() - t0)
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
body{{font-family:"Yu Gothic UI","Meiryo",sans-serif;max-width:1000px;margin:20px auto;padding:0 16px;background:#fafaf7;color:#222}}
h1{{font-size:2.2em;margin:.2em 0}} h2{{border-left:6px solid #2a7;padding-left:8px}}
.lead{{font-size:1.2em}}
form{{background:#fff;padding:16px;border-radius:10px;box-shadow:0 1px 4px #0002}}
fieldset{{border:none;margin:0 0 10px;padding:0}} legend{{font-weight:bold;font-size:1.1em}}
label{{display:inline-block;margin:4px 12px 4px 0;font-size:1.05em}}
input[type=number]{{width:5em;font-size:1.1em}}
.presets button{{font-size:1em;margin:3px;padding:6px 10px;border-radius:6px;border:1px solid #999;background:#f3f3f3;cursor:pointer}}
.go{{font-size:1.4em;padding:10px 40px;background:#2a7;color:#fff;border:none;border-radius:8px;cursor:pointer}}
.cards{{display:flex;gap:16px;margin-top:16px}} .card{{flex:1;padding:16px;border-radius:10px}}
.ok{{background:#e3f6e8;border:2px solid #2a7}} .ng{{background:#fde8e8;border:2px solid #d33}}
.big{{font-size:2.2em;font-weight:bold;margin:.2em 0}}
.msg{{font-size:1.15em;background:#fff8d8;padding:12px;border-radius:8px;margin-top:12px}}
#wait{{display:none;font-size:1.3em;color:#a60;margin-top:10px}}
.links a{{display:inline-block;margin:4px 8px 4px 0;font-size:1.05em}}
.note{{color:#666;font-size:.9em}}
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
<fieldset class="presets" style="margin-top:12px"><legend>② 家具をえらぶ</legend>
{furn_buttons}</fieldset>
<label>長さ <input type="number" name="fl" value="{fl}" min="20" max="300"> cm</label>
<label>幅 <input type="number" name="fw" value="{fw}" min="20" max="200"> cm</label>
<label>高さ <input type="number" name="fh" value="{fh}" min="20" max="200"> cm</label>
<p><button class="go" type="submit" name="go" value="1">判定する</button></p>
<p id="wait">計算中です… 新しい寸法だと20〜40秒ほどかかります</p>
</form>
{result}
<h2>くわしく見る</h2>
<div class="links">{links}</div>
<p class="note">運ぶ人は2人、体を円柱(半径20cm・身長170cm)で近似。家具を傾けられるのは55度まで、
持つ高さは10〜190cmと仮定しています。結果は研究中の試作品によるものです。</p>
<script>
function setv(o){{for(const k in o)document.querySelector('[name='+k+']').value=o[k];}}
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
                           result=result, links=links, fl=f"{fl:g}", fw=f"{fw:g}",
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
