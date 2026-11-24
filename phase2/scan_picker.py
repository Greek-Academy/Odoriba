"""
スキャンした空間で、出発点・目的地を自分で床に置き、家具の候補をまとめて
判定する画面。

    cd phase2
    python scan_picker.py <scan.obj>                 # http://localhost:8001 を開く
    python scan_picker.py --route                    # 検証用の合成経路(route_env)で
    python scan_picker.py <scan.obj> --furniture 180 60 40 --carriers 2

ブラウザに階ごとの間取り図(壁・床・階段。scan_floorplan が作る)が出る。
「出発点を置く」「目的地を置く」を選んで床か階段をクリックし(階が2つ以上
あればタブで切り替える)、候補(名前, 幅, 奥行, 高さ, 分かれば重さ)を
1行に1件ずつ書いて「判定する」を押すと、裏で掃引(1件あたり十数秒〜1分)を
回して、候補ごとの ○/△/× と理由を表で出す(scan_candidates と同じ判定)。
表の「3Dで見る」を押すと、その候補が一番狭い所でどう詰まるかを3D
(scan_viewer)で表示する。

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

import phase2_lstair as L
import scan_candidates as SC
import scan_floorplan as FP
import scan_plan as SP
import scan_space as SS
import scan_viewer as SV

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


def read_point(p):
    """床の1点 [x, y, z] を数値3つのリストにする(そう読めなければ None)。"""
    try:
        x, y, z = (float(v) for v in p)
    except (TypeError, ValueError):
        return None
    return [x, y, z] if np.isfinite([x, y, z]).all() else None


def can_view(row):
    """3Dで見せられる候補か(掃引していない候補は、見せる姿勢が無い)。"""
    return row["verdict"] != "unknown" and not row.get("too_heavy_flat")


def shown_row(row):
    """1件の結果を、画面の表に出す形にする(文言は scan_candidates のもの)。"""
    return dict(SC.row_parts(row), verdict=row["verdict"], can_view=can_view(row))


class App:
    """読み込んだスキャンと、判定ジョブの状態を持つ。

    間取り図は撮れた面すべてから作り(scan_floorplan)、空間(ScanSpace)は判定の
    ときに出発点の上を起点にして作る。起点を自動で選ぶと、つながっていない
    小さな部屋に入って、そこにしか置けなくなることがあるため。一度作った
    空間は、次の出発点がその中にあれば使い回す(作るのに約1分かかる)。
    """

    def __init__(self, mesh, num_carriers, seed=None, items_text=DEFAULT_ITEMS):
        self.mesh = mesh
        self.num_carriers = num_carriers
        self.fixed_seed = seed
        self.items_text = items_text
        # 判定中は carried_weight が L.CARRIER_PRESET を一時的に書き換えるので、
        # 既定値は起動時に控えたものを使う
        self.default_preset = L.CARRIER_PRESET
        self.space = None
        self.route = None     # いまの中心線の (出発点, 目的地)
        self.results = {}     # 判定ジョブ -> 3Dを作り直すのに要る値
        self.plans = FP.to_json(FP.build_plans(mesh))
        self.jobs = {}
        self.lock = threading.Lock()   # 中心線は ScanSpace に1つなので判定は1件ずつ

    def floor_json(self):
        return {"plans": self.plans,
                "kinds": {"nothing": FP.NOTHING, "floor": FP.FLOOR, "stair": FP.STAIR,
                          "other": FP.OTHER, "wall": FP.WALL},
                "carriers": self.num_carriers, "items": self.items_text,
                "presets": [{"key": k, "label": f"{SC.PRESET_TEXT[k]}(1人{kg:.0f}kgまで)"}
                            for k, kg in L.CARRIER_PRESETS.items()],
                "preset": self.default_preset,
                "legend": SC.LEGEND, "notes": list(SC.NOTES)}

    def _ensure_space(self, job, start):
        sp = self.space
        if sp is not None and float(sp.clearance_points([np.asarray(start) + [0, 0, 100]])[0]) > 0:
            return sp
        self.jobs[job] = {"state": "running", "message": "空間を作っています(約1分)..."}
        seed = self.fixed_seed or SS.seed_near(self.mesh, start)
        self.space = SS.ScanSpace(self.mesh, seed)
        self.route = None
        return self.space

    def _ensure_route(self, job, start, goal):
        """出発点・目的地の中心線を持つ空間を返す(同じ経路なら作り直さない)。"""
        space = self._ensure_space(job, start)
        key = (tuple(start), tuple(goal))
        if self.route != key:
            self.jobs[job] = {"state": "running", "message": "経路を作っています..."}
            space.build_centerline(start, goal)
            self.route = key
        return space

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
                space = self._ensure_route(job, start, goal)
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
                rows = SC.judge_candidates(space, items, num_carriers=self.num_carriers,
                                           carried_by=carried_by, log=lambda *_: None,
                                           on_row=on_row)
                length, leaked = space.centerline.s_total, space.leaked
            self.results[job] = {"rows": rows, "start": start, "goal": goal,
                                 "carried_by": carried_by}
            self.jobs[job] = {"state": "done", "rows": shown, "length": length,
                              "leaked": leaked}
        except Exception as e:
            traceback.print_exc()
            self.jobs[job] = {"state": "error", "message": str(e)}

    def submit_view(self, judged_job, index):
        """判定済みの候補1件の3Dを作るジョブを始める。(ジョブ番号, エラー文) を返す。"""
        res = self.results.get(judged_job)
        if res is None or not isinstance(index, int) or not 0 <= index < len(res["rows"]):
            return None, "その候補は見つかりません(もう一度判定してください)"
        if not can_view(res["rows"][index]):
            return None, "この候補は3Dで見せられません"
        job = uuid.uuid4().hex[:8]
        self.jobs[job] = {"state": "running", "message": "順番を待っています..."}
        threading.Thread(target=self._run_view, args=(job, res, res["rows"][index]),
                         daemon=True).start()
        return job, None

    def _run_view(self, job, res, row):
        try:
            a, b, up = row["carried_as"]
            with self.lock:
                space = self._ensure_route(job, res["start"], res["goal"])
                self.jobs[job] = {"state": "running",
                                  "message": "3Dを作っています(十数秒〜1分)..."}
                out = OUT_DIR / f"{job}.html"
                # 表と同じ向き・同じ重さで判定し直す(一括判定は姿勢を残していない)
                with SC.furniture_dims(a, b, up), \
                        SC.carried_weight(row.get("weight_kg"), res["carried_by"]):
                    # tilt_detail なしだと、重さで決まる上限の傾きも試すので、
                    # 表より少し甘い値が出る。表と同じ傾きの格子で判定する
                    result = SP.judge(space, num_carriers=self.num_carriers, rrt=False,
                                      tilt_detail=True, log=lambda *_: None)
                    SV.render_html(space, self.mesh, result, str(out),
                                   num_carriers=self.num_carriers,
                                   include_plotlyjs="/plotly.min.js",
                                   title=f"{row['name']} {SC.row_parts(row)['size']}")
            self.jobs[job] = {"state": "done", "url": f"/result/{job}.html"}
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
 #tabs button.on { background: #555; color: white; }
 .key { display: inline-block; width: 14px; height: 14px; border: 1px solid #999;
        vertical-align: middle; margin: 0 4px 0 12px; }
</style></head><body>
<h2>1. 出発点と目的地を床に置く</h2>
<div class="row">
 <button id="mStart" class="on" onclick="setMode('start')">● 出発点を置く</button>
 <button id="mGoal" onclick="setMode('goal')">■ 目的地を置く</button>
</div>
<div class="row" id="tabs"></div>
<div class="note" id="info"></div>
<div id="plan" style="width:100%;height:560px"></div>
<div class="note">
 <span class="key" style="background:#333;margin-left:0"></span>壁
 <span class="key" style="background:#dcdcdc"></span>床(置ける)
 <span class="key" style="background:#8fbde6"></span>階段(置ける)
 <span class="key" style="background:#f1e9d8"></span>家具など(置けない)
 <span class="key" style="background:#fff"></span>撮れていない所
</div>
<div class="row" id="placed"></div>
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
<div id="viewStatus" class="row"></div>
<iframe id="result"></iframe>
<div class="note" id="notes"></div>
<script>
let F = null, mode = 'start', pts = {start: null, goal: null};
let cur = 0, placedOn = {start: null, goal: null};
const NAMES = {start: '出発点', goal: '目的地'};
// マスの種類(何もない・床・階段・家具など・壁)の色。値は 0〜4 で、境目を 0.2 刻みに置く
const COLORS = ['#ffffff', '#dcdcdc', '#8fbde6', '#f1e9d8', '#333333'];
const SCALE = COLORS.flatMap((c, i) => [[i / 5, c], [(i + 1) / 5, c]]);
function setMode(m) {
  mode = m;
  document.getElementById('mStart').className = m === 'start' ? 'on' : '';
  document.getElementById('mGoal').className = m === 'goal' ? 'on' : '';
}
function setFloor(i) {
  cur = i;
  const tabs = document.getElementById('tabs');
  [...tabs.children].forEach((b, j) => { b.className = j === i ? 'on' : ''; });
  redraw();
}
function redraw() {
  const P = F.plans, f = P.floors[cur];
  const data = [{type: 'heatmap', z: f.kind, x0: P.x0 + P.cell / 2, dx: P.cell, y0: P.y0 + P.cell / 2, dy: P.cell,
    zmin: -0.5, zmax: 4.5, colorscale: SCALE, showscale: false, hoverinfo: 'none'}];
  for (const [k, sym, col] of [['start', 'circle', '#27ae60'], ['goal', 'square', '#c0392b']]) {
    const p = pts[k];
    if (p && placedOn[k] === cur) data.push({x: [p[0]], y: [p[1]], mode: 'markers+text', type: 'scatter', text: [NAMES[k]], textposition: 'top center',
      textfont: {size: 15, color: col}, marker: {size: 18, symbol: sym, color: col, line: {color: 'black', width: 2}}, hoverinfo: 'skip'});
  }
  const axis = {showgrid: false, zeroline: false, showticklabels: false, ticks: ''};
  Plotly.react('plan', data, {xaxis: {...axis, constrain: 'domain'}, yaxis: {...axis, scaleanchor: 'x'},
    margin: {t: 10, b: 10, l: 10, r: 10}, showlegend: false, dragmode: 'pan', plot_bgcolor: '#fff'}, {scrollZoom: true, displayModeBar: false});
  // 置いた場所を言葉でも出す(別の階に置いた印は、いまの図には出ないため)
  const where = ['start', 'goal'].map(k => NAMES[k] + ': ' +
    (pts[k] ? (F.plans.floors.length > 1 ? F.plans.floors[placedOn[k]].name + 'に置いた' : '置いた') : 'まだ置いていない'));
  document.getElementById('placed').textContent = where.join(' / ');
}
function onClick(ev) {
  const p = ev.points[0];
  if (p.curveNumber !== 0) return;
  const [iy, ix] = p.pointIndex, f = F.plans.floors[cur];
  const z = f.z[iy][ix], kind = f.kind[iy][ix];
  if (z === null || (kind !== F.kinds.floor && kind !== F.kinds.stair)) {
    document.getElementById('placed').textContent = 'そこには置けません。床(灰色)か階段(青)の上をクリックしてください';
    return;
  }
  pts[mode] = [p.x, p.y, z];
  placedOn[mode] = cur;
  if (mode === 'start') setMode('goal');
  redraw();
}
function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
let judged = null, busy = false;
const LOST = 'サーバーと通信できませんでした(scan_picker.py が動いているか確かめてください)';
// body があれば POST、無ければ GET。通信に失敗したら null を返す
async function ask(path, body) {
  try {
    const r = await fetch(path, body ? {method: 'POST', body: JSON.stringify(body)} : undefined);
    return await r.json();
  } catch (e) { return null; }
}
function showRows(rows, done) {
  const t = document.getElementById('rows');
  t.replaceChildren();
  if (!rows || !rows.length) return;
  const head = el('tr');
  for (const h of ['', '候補', '大きさ', '結果', '補足', '']) head.appendChild(el('th', h));
  t.appendChild(head);
  rows.forEach((r, i) => {
    const tr = el('tr', undefined, r.verdict);
    tr.appendChild(el('td', r.mark, 'mark'));
    tr.appendChild(el('td', r.name));
    tr.appendChild(el('td', r.size));
    tr.appendChild(el('td', r.reason));
    tr.appendChild(el('td', r.extras.join(' / ')));
    const td = el('td');
    if (done && r.can_view) {
      const b = el('button', '3Dで見る');
      b.onclick = () => view(i, r.name);
      td.appendChild(b);
    }
    tr.appendChild(td);
    t.appendChild(tr);
  });
}
async function view(i, name) {
  if (busy) { viewStatus('計算中です。終わってから押してください'); return; }
  busy = true;
  const fr = document.getElementById('result');
  fr.style.display = 'none';
  const lost = () => { busy = false; viewStatus(LOST); };
  const r = await ask('/view', {job: judged, index: i});
  if (!r) { lost(); return; }
  if (r.error) { busy = false; viewStatus(r.error); return; }
  const t0 = Date.now();
  const poll = async () => {
    const s = await ask('/status?job=' + r.job);
    if (!s) { lost(); return; }
    if (s.state === 'running') { viewStatus(name + ': ' + s.message + ' ' + Math.round((Date.now() - t0) / 1000) + '秒'); setTimeout(poll, 1000); return; }
    busy = false;
    if (s.state === 'error') { viewStatus(name + ': 3Dを作れませんでした: ' + s.message); return; }
    viewStatus(name + ': 一番狭い所での様子(左=家具だけ、右=人が運ぶ)');
    fr.src = s.url; fr.style.display = 'block';
  };
  poll();
}
function viewStatus(t) { document.getElementById('viewStatus').textContent = t; }
async function judge() {
  if (!pts.start || !pts.goal) { status('出発点と目的地の両方を置いてください'); return; }
  if (busy) { status('計算中です。終わってから押してください'); return; }
  status('計算中...');
  viewStatus('');
  showRows([]);
  document.getElementById('notes').style.display = 'none';
  document.getElementById('result').style.display = 'none';
  const body = {start: pts.start, goal: pts.goal, items: document.getElementById('items').value,
    carried_by: document.getElementById('by').value};
  const r = await ask('/judge', body);
  if (!r) { status(LOST); return; }
  if (r.error) { status(r.error); return; }
  busy = true;
  const t0 = Date.now();
  const poll = async () => {
    const s = await ask('/status?job=' + r.job);
    if (!s) { busy = false; status(LOST); return; }
    showRows(s.rows, s.state === 'done');
    if (s.state === 'running') { status(s.message + ' ' + Math.round((Date.now() - t0) / 1000) + '秒'); setTimeout(poll, 1000); return; }
    busy = false;
    if (s.state === 'error') { status('判定できませんでした: ' + s.message); return; }
    judged = r.job;
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
    ' / スキャンから作った間取り図。床か階段をクリックして置く(ドラッグで動かす、ホイールで拡大)';
  if (!d.plans.floors.length) {
    document.getElementById('info').textContent = 'このスキャンから床が見つかりませんでした。床が写るように撮り直してください';
    return;
  }
  const tabs = document.getElementById('tabs');
  if (d.plans.floors.length > 1) d.plans.floors.forEach((f, i) => {
    const b = el('button', f.name);
    b.onclick = () => setFloor(i);
    tabs.appendChild(b);
  });
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
  setFloor(0);
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
            path = urlparse(self.path).path
            if path not in ("/judge", "/view"):
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            except (TypeError, ValueError):
                body = None
            if not isinstance(body, dict):
                self._json({"error": "要求が読めませんでした"})
                return
            if path == "/view":
                job = body.get("job")
                job, error = app.submit_view(job if isinstance(job, str) else None,
                                             body.get("index"))
                self._json({"error": error} if error else {"job": job})
                return
            items, error = read_candidates(str(body.get("items", "")))
            carried_by = body.get("carried_by") or app.default_preset
            if error is None and not (isinstance(carried_by, str)
                                      and carried_by in L.CARRIER_PRESETS):
                error = "誰が運ぶかの指定が正しくありません"
            start, goal = read_point(body.get("start")), read_point(body.get("goal"))
            if error is None and (start is None or goal is None):
                error = "出発点と目的地の両方を置いてください"
            if error:
                self._json({"error": error})
                return
            self._json({"job": app.submit(start, goal, items, carried_by)})

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
