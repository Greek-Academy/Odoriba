"""
スキャンした空間で、出発点・目的地を自分で床に置いて判定する画面。

    cd phase2
    python scan_picker.py <scan.obj>                 # http://localhost:8001 を開く
    python scan_picker.py --route                    # 検証用の合成経路(route_env)で
    python scan_picker.py <scan.obj> --furniture 180 60 40 --carriers 2

ブラウザに床を上から見た図(高さで色分け)が出る。「出発点を置く」「目的地を
置く」を選んで床をクリックし、「判定する」を押すと、裏で掃引(十数秒〜1分)を
回して、詰まった所を赤く示す3D(scan_viewer)を表示する。階段のように床が
上下に重なる所は、表示する高さの範囲を絞ってから置く。

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
import scan_plan as SP
import scan_space as SS
import scan_viewer as SV

PORT = 8001
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
OUT_DIR = Path(tempfile.mkdtemp(prefix="odoriba_picker_"))


class App:
    """読み込んだスキャンと、判定ジョブの状態を持つ。

    床の図は撮れた床(上向きの面)すべてから作り、空間(ScanSpace)は判定の
    ときに出発点の上を起点にして作る。起点を自動で選ぶと、つながっていない
    小さな部屋に入って、そこにしか置けなくなることがあるため。一度作った
    空間は、次の出発点がその中にあれば使い回す(作るのに約1分かかる)。
    """

    def __init__(self, mesh, num_carriers, seed=None):
        self.mesh = mesh
        self.num_carriers = num_carriers
        self.fixed_seed = seed
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
                "furniture": [L.FURN_L, L.FURN_W, L.FURN_H],
                "carriers": self.num_carriers}

    def _seed_near(self, start):
        """出発点の上で、スキャンの面から一番離れた点(塗りつぶしの起点)。"""
        base = np.asarray(start, dtype=float)
        offs = [(dx, dy, dz) for dx in (-30, 0, 30) for dy in (-30, 0, 30)
                for dz in (80, 100, 130)]
        cand = base + np.array(offs, dtype=float)
        _, dist, _ = trimesh.proximity.closest_point(self.mesh, cand)
        return tuple(float(v) for v in cand[int(np.argmax(dist))])

    def _ensure_space(self, job, start):
        sp = self.space
        if sp is not None and float(sp.clearance_points([np.asarray(start) + [0, 0, 100]])[0]) > 0:
            return sp
        self.jobs[job]["message"] = "空間を作っています(約1分)..."
        seed = self.fixed_seed or self._seed_near(start)
        self.space = SS.ScanSpace(self.mesh, seed)
        return self.space

    def submit(self, start, goal):
        job = uuid.uuid4().hex[:8]
        self.jobs[job] = {"state": "running", "message": "順番を待っています..."}
        threading.Thread(target=self._run, args=(job, start, goal), daemon=True).start()
        return job

    def _run(self, job, start, goal):
        try:
            with self.lock:
                space = self._ensure_space(job, start)
                self.jobs[job]["message"] = "計算中(十数秒〜1分)..."
                space.build_centerline(start, goal)
                result = SP.judge(space, num_carriers=self.num_carriers, rrt=False,
                                  log=lambda *_: None)
                out = OUT_DIR / f"{job}.html"
                SV.render_html(space, self.mesh, result, str(out),
                               num_carriers=self.num_carriers,
                               include_plotlyjs="/plotly.min.js")
            b, bh = result["bottleneck_furniture_only"], result["bottleneck"]
            self.jobs[job] = {"state": "done", "url": f"/result/{job}.html",
                              "furniture_only": b["capacity_cm"],
                              "with_carriers": bh["capacity_cm"],
                              "length": space.centerline.s_total,
                              "leaked": space.leaked}
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
</style></head><body>
<h2>出発点と目的地を床に置く</h2>
<div class="row">
 <button id="mStart" class="on" onclick="setMode('start')">● 出発点を置く</button>
 <button id="mGoal" onclick="setMode('goal')">■ 目的地を置く</button>
 <button onclick="judge()">判定する</button>
</div>
<div class="row">表示する床の高さ:
 <input id="zmin" type="number" step="10" style="width:70px"> 〜
 <input id="zmax" type="number" step="10" style="width:70px"> cm
 <button onclick="redraw()">絞る</button> <button onclick="resetZ()">すべて</button>
 <span class="note">階段のように床が上下に重なる所は、高さを絞ってから置く</span>
</div>
<div class="note" id="info"></div>
<div id="plan" style="width:100%;height:560px"></div>
<div id="status"></div>
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
async function judge() {
  if (!pts.start || !pts.goal) { status('出発点と目的地の両方を置いてください'); return; }
  status('計算中...(十数秒〜1分)');
  document.getElementById('result').style.display = 'none';
  const r = await fetch('/judge', {method: 'POST', body: JSON.stringify(pts)});
  const {job} = await r.json();
  const t0 = Date.now();
  const poll = async () => {
    const s = await (await fetch('/status?job=' + job)).json();
    if (s.state === 'running') { status(s.message + ' ' + Math.round((Date.now() - t0) / 1000) + '秒'); setTimeout(poll, 1000); return; }
    if (s.state === 'error') { status('判定できませんでした: ' + s.message); return; }
    const v = (c) => c >= 0 ? '余裕 ' + c.toFixed(1) + 'cm' : 'あと ' + (-c).toFixed(1) + 'cm 足りない';
    status('経路 ' + Math.round(s.length) + 'cm / 家具だけ: ' + v(s.furniture_only) + ' / ' + F.carriers + '人で運ぶ: ' + v(s.with_carriers) +
      (s.leaked ? ' (注意: 撮れていない壁から空間が漏れている。判定は信用できない)' : ''));
    const fr = document.getElementById('result');
    fr.src = s.url; fr.style.display = 'block';
  };
  poll();
}
function status(t) { document.getElementById('status').textContent = t; }
fetch('/floor').then(r => r.json()).then(d => {
  F = d;
  document.getElementById('info').textContent = '家具 ' + d.furniture.map(Math.round).join('×') + 'cm / 運ぶ人 ' + d.carriers + '人' +
    ' / 色のついた点が撮れた床。クリックで置く(段や家具の上面も含むので、歩く床の上に置く)';
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
            self._json({"job": app.submit(body["start"], body["goal"])})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", nargs="?", help="スキャンのOBJ/PLY(--route なら不要)")
    parser.add_argument("--route", action="store_true", help="検証用の合成経路で試す")
    parser.add_argument("--furniture", type=float, nargs=3, metavar=("L", "W", "H"))
    parser.add_argument("--carriers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="自由空間の中の1点(既定は出発点の上から選ぶ)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()
    if args.furniture:
        L.FURN_L, L.FURN_W = sorted(args.furniture[:2], reverse=True)
        L.FURN_H = args.furniture[2]

    if args.route:
        import mesh_env as M
        import route_env as R
        mesh = M.env_to_mesh(*R.build_route())
        mesh.invert()   # スキャンと同じ「面が自由空間側を向く」向きにする
    elif args.mesh:
        import scan_demo as SD
        mesh, _ = SD.load_scan_zup(args.mesh)
    else:
        parser.error("mesh を指定する(または --route)")
    app = App(mesh, args.carriers, seed=args.seed)
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
