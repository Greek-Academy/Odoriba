"""
スキャンした空間で、出発点・目的地を自分で床に置き、家具の候補をまとめて
判定する画面。

    cd phase2
    python scan_picker.py <scan.obj>                 # http://localhost:8001 を開く
    python scan_picker.py --route                    # 検証用の合成経路(route_env)で
    python scan_picker.py <scan.obj> --furniture 180 60 40 --carriers 2

ブラウザに床を上から見た図(高さで色分け)が出る。「出発点を置く」「目的地を
置く」を選んで床をクリックし、候補(名前, 幅, 奥行, 高さ, 分かれば重さ)を
1行に1件ずつ書いて「判定する」を押すと、裏で掃引(1件あたり十数秒〜1分)を
回して、候補ごとの ○/△/× と理由を表で出す(scan_candidates と同じ判定)。
階段のように床が上下に重なる所は、表示する高さの範囲を絞ってから置く。

計算は時間がかかるので、要求を受けたらすぐ返し(ジョブ番号)、ブラウザが
進み具合を問い合わせる形にしている(同期APIにしない、という方針どおり)。
標準ライブラリの http.server だけで動く。
"""

import argparse
import json
import tempfile
import threading
import traceback
import uuid
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import plotly
import trimesh

import phase2_lstair as L
import scan_candidates as SC
import scan_space as SS

PORT = 8001
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
OUT_DIR = Path(tempfile.mkdtemp(prefix="odoriba_picker_"))

# 1件ずつ掃引するので、候補が多いと待ち時間が長くなる
MAX_ITEMS = 12
# 入力欄に最初から入れておく例(名前, 幅, 奥行, 高さ, 重さ)
DEFAULT_ITEMS = "ソファ, 180, 85, 80\n本棚, 90, 30, 180\n冷蔵庫, 60, 65, 170, 60"


def read_candidates(text):
    """入力欄の文字列を候補にする。(候補, 利用者に見せるエラー文 or None) を返す。

    読めない行が1つでもあれば判定しない(黙って落とすと、候補が減ったことに
    気づけないため)。
    """
    items, skipped = SC.parse_text(text)
    if skipped:
        return [], ("読めない行があります(名前, 幅, 奥行, 高さ の順に書く): "
                    + " / ".join(skipped))
    if not items:
        return [], "候補を1件以上書いてください"
    if len(items) > MAX_ITEMS:
        return [], f"候補は{MAX_ITEMS}件までにしてください(いま{len(items)}件)"
    for item in items:
        if SC.size_error(item):
            return [], SC.size_error(item)
    return items, None


def shown_row(row):
    """1件の結果を、画面の表に出す形にする(文言は scan_candidates のもの)。"""
    return dict(SC.row_parts(row), verdict=row["verdict"])


class App:
    """読み込んだスキャンと、判定ジョブの状態を持つ。

    床の図は撮れた床(上向きの面)すべてから作り、空間(ScanSpace)は判定の
    ときに出発点の上を起点にして作る。起点を自動で選ぶと、つながっていない
    小さな部屋に入って、そこにしか置けなくなることがあるため。一度作った
    空間は、次の出発点がその中にあれば使い回す(作るのに約1分かかる)。
    """

    def __init__(self, mesh, num_carriers, seed=None, items_text=DEFAULT_ITEMS):
        self.mesh = mesh
        self.num_carriers = num_carriers
        self.fixed_seed = seed
        self.items_text = items_text
        self.space = None
        self.floor = self._floor_points()
        self.jobs = {}
        self.lock = threading.Lock()   # 中心線は ScanSpace に1つなので判定は1件ずつ

    def _floor_points(self, n=30000):
        """撮れた床(上向きの面)の上に、面積に比例して点を撒く(床の図用)。"""
        nz = SS._floor_facing_z(self.mesh)
        faces = np.flatnonzero(nz > 0.9)
        sub = self.mesh.submesh([faces], append=True)
        pts, _ = trimesh.sample.sample_surface(sub, n, seed=0)
        return pts

    def floor_json(self):
        f = self.floor
        return {"x": np.round(f[:, 0], 1).tolist(), "y": np.round(f[:, 1], 1).tolist(),
                "z": np.round(f[:, 2], 1).tolist(),
                "carriers": self.num_carriers, "items": self.items_text,
                "presets": [{"key": k, "label": f"{SC.PRESET_TEXT[k]}(1人{kg:.0f}kgまで)"}
                            for k, kg in L.CARRIER_PRESETS.items()],
                "preset": L.CARRIER_PRESET,
                "legend": SC.LEGEND, "notes": list(SC.NOTES)}

    def _ensure_space(self, job, start):
        sp = self.space
        if sp is not None and float(sp.clearance_points([np.asarray(start) + [0, 0, 100]])[0]) > 0:
            return sp
        self.jobs[job]["message"] = "空間を作っています(約1分)..."
        seed = self.fixed_seed or SS.seed_near(self.mesh, start)
        self.space = SS.ScanSpace(self.mesh, seed)
        return self.space

    def submit(self, start, goal, items, carried_by):
        job = uuid.uuid4().hex[:8]
        self.jobs[job] = {"state": "running", "message": "順番を待っています..."}
        threading.Thread(target=self._run, args=(job, start, goal, items, carried_by),
                         daemon=True).start()
        return job

    def _run(self, job, start, goal, items, carried_by):
        try:
            # 候補ごとに家具の寸法(phase2_lstair の変数)を差し替えて判定するので、
            # 必ず lock の中で1件ずつ動かす(並行に動かすと寸法が混ざる)
            with self.lock:
                space = self._ensure_space(job, start)
                self.jobs[job] = {"state": "running", "message": "経路を作っています..."}
                space.build_centerline(start, goal)
                shown = []

                def progress():
                    # 状態は丸ごと差し替える(問い合わせ側が読んでいる途中で変えない)
                    self.jobs[job] = {
                        "state": "running", "rows": list(shown),
                        "message": f"判定中 {len(shown)}/{len(items)} 件"
                                   "(1件あたり十数秒〜1分)..."}

                def on_row(row):
                    shown.append(shown_row(row))
                    progress()

                progress()
                SC.judge_candidates(space, items, num_carriers=self.num_carriers,
                                    carried_by=carried_by, log=lambda *_: None,
                                    on_row=on_row)
                length, leaked = space.centerline.s_total, space.leaked
            self.jobs[job] = {"state": "done", "rows": shown, "length": length,
                              "leaked": leaked}
        except Exception as e:
            traceback.print_exc()
            self.jobs[job] = {"state": "error", "message": str(e)}


PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>Odoriba: 出発点と目的地を置く</title>
<script src="/plotly.min.js"></script>
<style>
 body { font-family: sans-serif; margin: 16px; color: #333; }
 button { font-size: 15px; padding: 6px 14px; margin-right: 6px; }
 button.on { background: #2c6fbb; color: white; }
 #status { margin: 10px 0; font-weight: bold; }
 #result { width: 100%; height: 760px; border: 1px solid #ccc; display: none; }
 .note { color: #666; font-size: 13px; }
 .row { margin: 8px 0; }
 textarea { width: 420px; height: 110px; font-size: 14px; vertical-align: top; }
 select { font-size: 15px; }
 table { border-collapse: collapse; margin: 8px 0; }
 th, td { border: 1px solid #ccc; padding: 5px 10px; text-align: left; font-size: 15px; }
 th { background: #f3f3f3; }
 td.mark { font-size: 22px; text-align: center; font-weight: bold; }
 tr.ok td.mark { color: #27ae60; } tr.close td.mark { color: #d68910; }
 tr.ng td.mark { color: #c0392b; } tr.unknown td.mark { color: #888; }
 #notes { display: none; }
</style></head><body>
<h2>1. 出発点と目的地を床に置く</h2>
<div class="row">
 <button id="mStart" class="on" onclick="setMode('start')">● 出発点を置く</button>
 <button id="mGoal" onclick="setMode('goal')">■ 目的地を置く</button>
</div>
<div class="row">表示する床の高さ:
 <input id="zmin" type="number" step="10" style="width:70px"> 〜
 <input id="zmax" type="number" step="10" style="width:70px"> cm
 <button onclick="redraw()">絞る</button> <button onclick="resetZ()">すべて</button>
 <span class="note">階段のように床が上下に重なる所は、高さを絞ってから置く</span>
</div>
<div class="note" id="info"></div>
<div id="plan" style="width:100%;height:560px"></div>
<h2>2. 運びたい家具の候補を書く</h2>
<div class="row">
 <textarea id="items"></textarea>
 <span class="note">1行に1件: 名前, 幅, 奥行, 高さ(cm)。重さ(kg)が分かれば5つ目に書く</span>
</div>
<div class="row">重さを書いた候補を運ぶのは: <select id="by"></select>
 <button onclick="judge()">判定する</button>
</div>
<div id="status"></div>
<table id="rows"></table>
<div class="note" id="notes"></div>
<iframe id="result"></iframe>
<script>
let F = null, mode = 'start', pts = {start: null, goal: null};
function setMode(m) {
  mode = m;
  document.getElementById('mStart').className = m === 'start' ? 'on' : '';
  document.getElementById('mGoal').className = m === 'goal' ? 'on' : '';
}
function zrange() {
  return [parseFloat(document.getElementById('zmin').value), parseFloat(document.getElementById('zmax').value)];
}
function resetZ() {
  document.getElementById('zmin').value = Math.floor(Math.min(...F.z));
  document.getElementById('zmax').value = Math.ceil(Math.max(...F.z));
  redraw();
}
function redraw() {
  const [lo, hi] = zrange();
  const x = [], y = [], z = [];
  for (let i = 0; i < F.x.length; i++) if (F.z[i] >= lo && F.z[i] <= hi) { x.push(F.x[i]); y.push(F.y[i]); z.push(F.z[i]); }
  const data = [{x, y, mode: 'markers', type: 'scattergl', customdata: z,
    marker: {size: 5, color: z, colorscale: 'Viridis', colorbar: {title: '床の高さ(cm)'}},
    hovertemplate: 'x=%{x} y=%{y} 高さ=%{customdata}cm<extra></extra>'}];
  for (const [k, sym, col, name] of [['start', 'circle', '#27ae60', '出発点'], ['goal', 'square', '#c0392b', '目的地']]) {
    const p = pts[k];
    if (p) data.push({x: [p[0]], y: [p[1]], mode: 'markers+text', type: 'scatter', text: [name], textposition: 'top center',
      marker: {size: 18, symbol: sym, color: col, line: {color: 'black', width: 2}}, hoverinfo: 'skip'});
  }
  Plotly.react('plan', data, {xaxis: {title: 'x (cm)', scaleanchor: 'y'}, yaxis: {title: 'y (cm)'},
    margin: {t: 20}, showlegend: false, dragmode: 'pan'}, {scrollZoom: true});
}
function onClick(ev) {
  const p = ev.points[0];
  if (p.curveNumber !== 0) return;
  pts[mode] = [p.x, p.y, p.customdata];
  if (mode === 'start') setMode('goal');
  redraw();
}
function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
function showRows(rows) {
  const t = document.getElementById('rows');
  t.replaceChildren();
  if (!rows || !rows.length) return;
  const head = el('tr');
  for (const h of ['', '候補', '大きさ', '結果', '補足']) head.appendChild(el('th', h));
  t.appendChild(head);
  for (const r of rows) {
    const tr = el('tr', undefined, r.verdict);
    tr.appendChild(el('td', r.mark, 'mark'));
    tr.appendChild(el('td', r.name));
    tr.appendChild(el('td', r.size));
    tr.appendChild(el('td', r.reason));
    tr.appendChild(el('td', r.extras.join(' / ')));
    t.appendChild(tr);
  }
}
async function judge() {
  if (!pts.start || !pts.goal) { status('出発点と目的地の両方を置いてください'); return; }
  status('計算中...');
  showRows([]);
  document.getElementById('notes').style.display = 'none';
  document.getElementById('result').style.display = 'none';
  const body = {start: pts.start, goal: pts.goal, items: document.getElementById('items').value,
    carried_by: document.getElementById('by').value};
  const r = await (await fetch('/judge', {method: 'POST', body: JSON.stringify(body)})).json();
  if (r.error) { status(r.error); return; }
  const t0 = Date.now();
  const poll = async () => {
    const s = await (await fetch('/status?job=' + r.job)).json();
    showRows(s.rows);
    if (s.state === 'running') { status(s.message + ' ' + Math.round((Date.now() - t0) / 1000) + '秒'); setTimeout(poll, 1000); return; }
    if (s.state === 'error') { status('判定できませんでした: ' + s.message); return; }
    status('経路 ' + Math.round(s.length) + 'cm を ' + F.carriers + '人で運ぶ場合' +
      (s.leaked ? ' (注意: 撮れていない壁から空間が漏れている。判定は信用できない)' : ''));
    document.getElementById('notes').style.display = 'block';
  };
  poll();
}
function status(t) { document.getElementById('status').textContent = t; }
fetch('/floor').then(r => r.json()).then(d => {
  F = d;
  document.getElementById('info').textContent = '運ぶ人 ' + d.carriers + '人' +
    ' / 色のついた点が撮れた床。クリックで置く(段や家具の上面も含むので、歩く床の上に置く)';
  document.getElementById('items').value = d.items;
  const by = document.getElementById('by');
  for (const p of d.presets) { const o = el('option', p.label); o.value = p.key; by.appendChild(o); }
  by.value = d.preset;
  const notes = document.getElementById('notes');
  notes.appendChild(el('div', d.legend));
  notes.appendChild(el('div', '注意:'));
  const ul = el('ul');
  for (const n of d.notes) ul.appendChild(el('li', n));
  notes.appendChild(ul);
  resetZ();
  document.getElementById('plan').on('plotly_click', onClick);
});
</script></body></html>
"""


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, data, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj):
            self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/plotly.min.js":
                self._send(PLOTLY_JS.read_bytes(), "application/javascript")
            elif u.path == "/floor":
                self._json(app.floor_json())
            elif u.path == "/status":
                job = parse_qs(u.query).get("job", [""])[0]
                self._json(app.jobs.get(job, {"state": "error", "message": "不明なジョブ"}))
            elif u.path.startswith("/result/") and u.path.endswith(".html"):
                f = OUT_DIR / Path(u.path).name
                if f.exists():
                    self._send(f.read_bytes(), "text/html; charset=utf-8")
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/judge":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            items, error = read_candidates(str(body.get("items", "")))
            carried_by = body.get("carried_by") or L.CARRIER_PRESET
            if error is None and carried_by not in L.CARRIER_PRESETS:
                error = "誰が運ぶかの指定が正しくありません"
            if error:
                self._json({"error": error})
                return
            self._json({"job": app.submit(body["start"], body["goal"], items, carried_by)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", nargs="?", help="スキャンのOBJ/PLY(--route なら不要)")
    parser.add_argument("--route", action="store_true", help="検証用の合成経路で試す")
    parser.add_argument("--furniture", type=float, nargs=3, metavar=("W", "D", "H"),
                        help="入力欄に最初から入れておく候補の 幅 奥行 高さ(cm)")
    parser.add_argument("--carriers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="自由空間の中の1点(既定は出発点の上から選ぶ)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()
    items_text = DEFAULT_ITEMS
    if args.furniture:
        items_text = "家具, " + ", ".join(f"{x:g}" for x in args.furniture)

    if args.route:
        import mesh_env as M
        import route_env as R
        mesh = M.env_to_mesh(*R.build_route())
        mesh.invert()   # スキャンと同じ「面が自由空間側を向く」向きにする
    elif args.mesh:
        import scan_ingest as SI
        mesh, _ = SI.load_scan_zup(args.mesh)
    else:
        parser.error("mesh を指定する(または --route)")
    app = App(mesh, args.carriers, seed=args.seed, items_text=items_text)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://localhost:{args.port}"
    print(f"[picker] {url} で起動しました(Ctrl+Cで終了)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
