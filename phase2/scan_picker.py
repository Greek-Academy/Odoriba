"""
スキャンした空間で、出発点・目的地を自分で床に置き、家具の候補をまとめて
判定する画面。

    cd phase2
    python scan_picker.py <scan.obj>                 # http://localhost:8001 を開く
    python scan_picker.py --route                    # 検証用の合成経路(route_env)で
    python scan_picker.py <scan.obj> --furniture 180 60 40 --carriers 2

初回だけ、立体表示の部品を取ってくる: python fetch_viewer.py

ブラウザに、スキャンがそのまま立体で出る(写真つき。Babylon.js)。ドラッグで
回し、「出発点を置く」「目的地を置く」を選んで床をクリックするとピンが立つ。
以前は床の点を高さで色分けした図や、階ごとの間取り図に描き直していたが、
実際の家と見比べる手がかりが無く、使えなかった(decisions.md 2026-11-24)。
候補(名前, 幅, 奥行, 高さ, 分かれば重さ)を1行に1件ずつ書いて「判定する」を押すと、裏で掃引(1件あたり十数秒〜1分)を
回して、候補ごとの ○/△/× と理由を表で出す(scan_candidates と同じ判定)。
表の「立体で見る」を押すと、同じ立体表示の中に、経路の線・一番狭い所での家具と
運ぶ人・当たる所の印を重ね、そこへ視点を寄せる(scan_scene)。経路上の位置を
動かして、家具が運ばれていく様子も見られる。

計算は時間がかかるので、要求を受けたらすぐ返し(ジョブ番号)、ブラウザが
進み具合を問い合わせる形にしている(同期APIにしない、という方針どおり)。
標準ライブラリの http.server だけで動く。
"""

import argparse
import atexit
import json
import shutil
import tempfile
import threading
import traceback
import uuid
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote, unquote

import numpy as np

import fetch_viewer
import phase2_lstair as L
import scan_candidates as SC
import scan_plan as SP
import scan_scene as SCN
import scan_space as SS

PORT = 8001

# 立体表示に渡すファイルの種類(スキャンの形・材質・写真)
VIEW_TYPES = {".obj": "text/plain", ".mtl": "text/plain", ".jpg": "image/jpeg",
              ".jpeg": "image/jpeg", ".png": "image/png"}
VENDOR_TYPES = {".js": "application/javascript"}


def plain_name(name):
    """フォルダの直下に置いてよい、ただのファイル名なら返す(そうでなければ None)。

    最後の「/」より後ろを取り、ドライブ指定や区切りが残る名前は断る。Windows
    では folder / "D:x.obj" が別のドライブを指すので、名前だけを取っても
    フォルダの外に出られてしまう。
    """
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    if base in ("", ".", "..") or ":" in base:
        return None
    return base


def temp_dir():
    """終了時に消す一時フォルダを作る(展開したスキャンを残さない)。"""
    out = Path(tempfile.mkdtemp(prefix="odoriba_scan_"))
    atexit.register(shutil.rmtree, out, ignore_errors=True)
    return out


def view_of_scan(path, to_zup):
    """スキャンの元ファイルを、立体表示に渡せる形にする。

    戻り値は {"dir": 配るフォルダ, "file": OBJの名前, "to_zup": 行列}。zip なら
    中の OBJ・MTL・写真を一時フォルダに出す。OBJ を直接渡されたときは、同じ
    フォルダの MTL・写真をそのまま配る。to_zup は、元のファイルの座標を判定側の
    座標(上がZ・cm)に直す行列(scan_ingest.load_scan_zup の info["to_zup"])。
    OBJ でないもの(PLY など)は None を返す(呼び出し側で、読み込んだ形だけを出す)。
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(path) as z:
            # 判定側(scan_ingest.load_mesh)は、中の最初の .obj/.ply を読む。同じものを出す
            meshes = [n for n in z.namelist() if n.lower().endswith((".obj", ".ply"))]
            first = plain_name(meshes[0]) if meshes else None
            if first is None or not first.lower().endswith(".obj"):
                return None
            out = temp_dir()
            for name in z.namelist():
                base = plain_name(name)
                if base is None or Path(base).suffix.lower() not in VIEW_TYPES:
                    continue
                if base.lower().endswith(".obj") and name != meshes[0]:
                    continue
                with z.open(name) as src, open(out / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        return {"dir": out, "file": first, "to_zup": to_zup}
    if path.suffix.lower() != ".obj":
        return None
    return {"dir": path.resolve().parent, "file": path.name, "to_zup": to_zup}


def view_of_mesh(mesh):
    """判定側の座標(上がZ・cm)のメッシュを、立体表示に渡せる形にする。

    写真の無い合成経路や、OBJ でないスキャン(PLY など)用。立体表示は上がYなので、X軸まわりに-90度回して
    書き出し、行列で元に戻す。
    """
    import trimesh
    back = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    m = mesh.copy()
    m.apply_transform(np.linalg.inv(back))
    out = temp_dir()
    m.export(out / "scan.obj", include_normals=True)
    return {"dir": out, "file": "scan.obj", "to_zup": back.tolist()}


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


def margin_text(cm):
    """余裕(cm)を「余裕◯cm / あと◯cm足りない」の言葉にする。"""
    return f"余裕{cm:.0f}cm" if cm >= 0 else f"あと{-cm:.0f}cm足りない"


def shown_row(row):
    """1件の結果を、画面の表に出す形にする(文言は scan_candidates のもの)。"""
    return dict(SC.row_parts(row), verdict=row["verdict"], can_view=can_view(row))


class App:
    """読み込んだスキャンと、判定ジョブの状態を持つ。

    画面にはスキャンの元ファイルをそのまま立体で出し(view)、空間(ScanSpace)は
    判定のときに出発点の上を起点にして作る。起点を自動で選ぶと、つながっていない
    小さな部屋に入って、そこにしか置けなくなることがあるため。一度作った
    空間は、次の出発点がその中にあれば使い回す(作るのに約1分かかる)。
    """

    def __init__(self, mesh, view, num_carriers, seed=None, items_text=DEFAULT_ITEMS):
        self.mesh = mesh
        self.view = view      # 立体表示に渡すファイルと行列(view_of_scan / view_of_mesh)
        self.num_carriers = num_carriers
        self.fixed_seed = seed
        self.items_text = items_text
        # 判定中は carried_weight が L.CARRIER_PRESET を一時的に書き換えるので、
        # 既定値は起動時に控えたものを使う
        self.default_preset = L.CARRIER_PRESET
        self.space = None
        self.route = None     # いまの中心線の (出発点, 目的地)
        self.results = {}     # 判定ジョブ -> 3Dを作り直すのに要る値
        self.jobs = {}
        self.lock = threading.Lock()   # 中心線は ScanSpace に1つなので判定は1件ずつ

    def floor_json(self):
        return {"scan": "/scan/" + quote(self.view["file"]),
                "to_zup": self.view["to_zup"],
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
        """判定済みの候補1件を立体表示に重ねる形にするジョブを始める。
        (ジョブ番号, エラー文) を返す。"""
        res = self.results.get(judged_job)
        if res is None or not isinstance(index, int) or not 0 <= index < len(res["rows"]):
            return None, "その候補は見つかりません(もう一度判定してください)"
        if not can_view(res["rows"][index]):
            return None, "この候補は立体で見せられません"
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
                                  "message": "立体表示を作っています(十数秒〜1分)..."}
                # 表と同じ向き・同じ重さで判定し直す(一括判定は姿勢を残していない)
                with SC.furniture_dims(a, b, up), \
                        SC.carried_weight(row.get("weight_kg"), res["carried_by"]):
                    # tilt_detail なしだと、重さで決まる上限の傾きも試すので、
                    # 表より少し甘い値が出る。表と同じ傾きの格子で判定する
                    result = SP.judge(space, num_carriers=self.num_carriers, rrt=False,
                                      tilt_detail=True, log=lambda *_: None)
                    scene = SCN.result_scene(space, result, self.num_carriers,
                                             np.linalg.inv(np.array(self.view["to_zup"])))
            parts = SC.row_parts(row)
            scene.update(name=row["name"], size=parts["size"], reason=parts["reason"],
                         furniture_only=margin_text(scene["furniture_only_cm"]))
            self.jobs[job] = {"state": "done", "scene": scene}
        except Exception as e:
            traceback.print_exc()
            self.jobs[job] = {"state": "error", "message": str(e)}


PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>Odoriba: 出発点と目的地を置く</title>
<script src="/vendor/babylon.js"></script>
<script src="/vendor/babylonjs.loaders.min.js"></script>
<style>
 body { font-family: sans-serif; margin: 16px; color: #333; }
 button { font-size: 15px; padding: 6px 14px; margin-right: 6px; }
 button.on { background: #2c6fbb; color: white; }
 #status { margin: 10px 0; font-weight: bold; }
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
 #view { width: 100%; height: 600px; border: 1px solid #ccc; outline: none; touch-action: none; display: block; }
 #stage { position: relative; overflow: hidden; }
 #tag { position: absolute; display: none; transform: translate(-50%, -140%); pointer-events: none;
        background: rgba(255,255,255,0.93); border: 2px solid #c0392b; border-radius: 6px;
        padding: 4px 10px; font-weight: bold; font-size: 15px; white-space: nowrap; }
 #tag.ok { border-color: #27ae60; }
 #play { display: none; background: #f6f8fa; border: 1px solid #ccc; padding: 8px 10px; }
 #pos { width: 420px; vertical-align: middle; }
 #cut { width: 320px; vertical-align: middle; }
</style></head><body>
<h2>1. 出発点と目的地のピンを床に立てる</h2>
<div class="row">
 <button id="mStart" class="on" onclick="setMode('start')">● 出発点を置く</button>
 <button id="mGoal" onclick="setMode('goal')">■ 目的地を置く</button>
</div>
<div class="note" id="info"></div>
<div id="stage"><canvas id="view"></canvas><div id="tag"></div></div>
<div class="row" id="play">
 <div id="headline" style="font-weight: bold"></div>
 <div class="row">
  <button id="playBtn" onclick="togglePlay()">▶ 運んでみる</button>
  <button onclick="goWorst()">一番狭い所へ</button>
  <input id="pos" type="range" min="0" max="0" step="0.02" value="0">
  <span id="posText"></span>
  <button onclick="clearResult()">重ねた結果を消す</button>
 </div>
 <div class="note">線の色: 緑=余裕あり / 黄=5cm未満 / 赤=足りない / 紫=持てる姿勢が無い。家具と人は、各位置で試した中で一番余裕のある姿勢。位置(15cmごと)の間は、動きが見やすいよう形をなめらかにつないでいるだけで、その途中で引っかからないかは確かめていない</div>
</div>
<div class="row">この高さから上を隠す: <input id="cut" type="range" min="0" max="1000" value="1000">
 <span class="note" id="cutText">隠していない</span>
 <span class="note">(2階の床で1階が見えないときに下げる)</span>
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
<div class="note" id="notes"></div>
<script>
let F = null, mode = 'start', pts = {start: null, goal: null};
let scene = null, engine = null, cam = null, bounds = null, unit = 1, pins = {start: null, goal: null};
let R = null;       // 重ねて表示している結果(無ければ null)
let faceSign = 1;   // 面の頂点の並びから出した向きに掛けて、「部屋の内側を向く向き」にする符号
const NAMES = {start: '出発点', goal: '目的地'};
const PIN_COLORS = {start: '#27ae60', goal: '#c0392b'};
function setMode(m) {
  mode = m;
  document.getElementById('mStart').className = m === 'start' ? 'on' : '';
  document.getElementById('mGoal').className = m === 'goal' ? 'on' : '';
}
// 元のファイルの座標 -> 判定側の座標(上がZ・cm)
function toZup(v) {
  return F.to_zup.slice(0, 3).map(r => r[0] * v.x + r[1] * v.y + r[2] * v.z + r[3]);
}
// 向き(法線)を判定側の座標にしたときの、上向きの成分(-1〜1)
function upOf(n) {
  const r = F.to_zup, v = [0, 1, 2].map(i => r[i][0] * n.x + r[i][1] * n.y + r[i][2] * n.z);
  return v[2] / Math.hypot(v[0], v[1], v[2]);
}
function showPlaced(msg) {
  const where = ['start', 'goal'].map(k => NAMES[k] + ': ' +
    (pts[k] ? '置いた(高さ ' + Math.round(pts[k][2]) + 'cm)' : 'まだ置いていない'));
  document.getElementById('placed').textContent = (msg ? msg + ' / ' : '') + where.join(' / ');
}
function putPin(k, p) {
  if (pins[k]) pins[k].dispose(false, true);   // 材質も一緒に消す(置き直すたびに溜めない)
  const h = 60 / unit, r = 9 / unit;   // ピンの長さ60cm・頭の半径9cmを、ファイルの単位にする
  const mat = new BABYLON.StandardMaterial('pin-' + k, scene);
  mat.emissiveColor = BABYLON.Color3.FromHexString(PIN_COLORS[k]);
  mat.disableLighting = true;
  const stem = BABYLON.MeshBuilder.CreateCylinder('stem', {height: h, diameterTop: r * 0.5, diameterBottom: 0}, scene);
  stem.position.y = h / 2;
  const head = BABYLON.MeshBuilder.CreateSphere('head', {diameter: r * 2}, scene);
  head.position.y = h;
  const pin = BABYLON.Mesh.MergeMeshes([stem, head], true);
  pin.material = mat;
  pin.position.copyFrom(p);
  pin.isPickable = false;
  pin.renderingGroupId = 1;   // 壁の向こうにあっても見えるよう、最後に描く
  pins[k] = pin;
}
// 面の向き(頂点の並びから出す。長さは面積の2倍)。部品の getNormal は、頂点に
// 付いた向きに合わせて符号を変えるので、角の丸められた向きでは当てにならない
function rawNormal(mesh, face) {
  const idx = mesh.getIndices(), pos = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
  const v = k => { const i = idx[face * 3 + k] * 3; return new BABYLON.Vector3(pos[i], pos[i + 1], pos[i + 2]); };
  const a = v(0), n = BABYLON.Vector3.Cross(v(1).subtract(a), v(2).subtract(a));
  return BABYLON.Vector3.TransformNormal(n, mesh.getWorldMatrix());
}
// 一番低い所(床があるはず)の水平な面が、上向きと下向きのどちらで多く出るかを見て、
// 「部屋の内側を向く向き」の符号を決める(scan_space._floor_facing_z と同じ考え方)
function findFaceSign(meshes) {
  let up = 0, down = 0;
  const top = bounds.min.y + 0.1 * (bounds.max.y - bounds.min.y);
  for (const m of meshes) {
    const idx = m.getIndices(), pos = m.getVerticesData(BABYLON.VertexBuffer.PositionKind);
    for (let f = 0; f < idx.length / 3; f++) {
      const y = (pos[idx[f * 3] * 3 + 1] + pos[idx[f * 3 + 1] * 3 + 1] + pos[idx[f * 3 + 2] * 3 + 1]) / 3;
      if (y > top) continue;
      const n = rawNormal(m, f), len = n.length();
      if (len === 0) continue;
      if (n.y / len > 0.9) up += len; else if (n.y / len < -0.9) down += len;
    }
  }
  return down > up ? -1 : 1;
}
// 画面の (x, y) で、見えている面のうち一番手前のものを返す。
// 当たり判定は一番手前の面しか返さないので、裏から見ている面(描かれていない)や
// 隠している高さより上の面に当たったら、その少し先から探し直す
function pickVisible(x, y) {
  const ray = scene.createPickingRay(x, y, BABYLON.Matrix.Identity(), scene.activeCamera);
  const cutY = scene.clipPlane ? -scene.clipPlane.d : Infinity;
  const step = bounds.max.subtract(bounds.min).length() * 1e-4;
  for (let i = 0; i < 80; i++) {
    const h = scene.pickWithRay(ray, m => m.isPickable);
    if (!h || !h.hit) return null;
    const n = rawNormal(h.pickedMesh, h.faceId).scale(faceSign);
    if (h.pickedPoint.y <= cutY && BABYLON.Vector3.Dot(n, ray.direction) < 0) return {point: h.pickedPoint, normal: n};
    ray.origin = h.pickedPoint.add(ray.direction.scale(step));
  }
  return null;
}
function onTap(x, y) {
  const hit = pickVisible(x, y);
  if (!hit) { showPlaced('何もない所です。床の上をクリックしてください'); return; }
  if (upOf(hit.normal) < 0.7) { showPlaced('そこは壁か天井です。床の上をクリックしてください'); return; }
  pts[mode] = toZup(hit.point).map(v => Math.round(v * 10) / 10);
  putPin(mode, hit.point);
  if (mode === 'start') setMode('goal');
  showPlaced();
}
function setCut() {
  const s = document.getElementById('cut'), t = s.value / s.max;
  if (t >= 1) { scene.clipPlane = null; document.getElementById('cutText').textContent = '隠していない'; return; }
  const y = bounds.min.y + (bounds.max.y - bounds.min.y) * t;
  scene.clipPlane = new BABYLON.Plane(0, 1, 0, -y);   // y より上を描かない
  document.getElementById('cutText').textContent = '高さ ' + Math.round(toZup({x: 0, y: y, z: 0})[2]) + 'cm より上を隠している';
}
async function startView() {
  const info = document.getElementById('info');
  if (typeof BABYLON === 'undefined') {
    info.textContent = '立体表示の部品がありません。phase2 で「python fetch_viewer.py」を一度実行してから、scan_picker を立て直してください';
    return;
  }
  const canvas = document.getElementById('view');
  engine = new BABYLON.Engine(canvas, true, {preserveDrawingBuffer: true});
  scene = new BABYLON.Scene(engine);
  scene.useRightHandedSystem = true;   // ファイルの座標をそのまま使う(左右を反転させない)
  scene.clearColor = new BABYLON.Color4(0.95, 0.95, 0.95, 1);
  new BABYLON.HemisphericLight('light', new BABYLON.Vector3(0.3, 1, 0.2), scene);
  info.textContent = 'スキャンを読み込んでいます...';
  const cut = F.scan.lastIndexOf('/') + 1;
  const t0 = Date.now();
  let res;
  try {
    res = await BABYLON.SceneLoader.ImportMeshAsync('', F.scan.slice(0, cut), F.scan.slice(cut), scene, ev => {
      if (ev.lengthComputable) info.textContent = 'スキャンを読み込んでいます... ' + Math.round(100 * ev.loaded / ev.total) + '%';
    });
  } catch (e) {
    info.textContent = 'スキャンを読み込めませんでした: ' + e.message;
    return;
  }
  const meshes = res.meshes.filter(m => m.getTotalVertices() > 0);
  let min = null, max = null;
  for (const m of meshes) {
    m.computeWorldMatrix(true);
    const b = m.getBoundingInfo().boundingBox;
    min = min ? BABYLON.Vector3.Minimize(min, b.minimumWorld) : b.minimumWorld.clone();
    max = max ? BABYLON.Vector3.Maximize(max, b.maximumWorld) : b.maximumWorld.clone();
  }
  bounds = {min, max};
  // 部屋の内側を向いた面だけ描く。外から見ると、手前の壁や天井が透けて中が見える
  // (サンドボックスと同じ見え方)。この設定(右手系)で読み込むと、ふつうのスキャン
  // (面が撮った人の側を向く)は、頂点の並びから出した向きが内側と逆(符号 -1)になり、
  // そのまま内側の面が描かれる。符号が逆のファイル(面が外を向いている)は、面を
  // 裏返して同じ形にそろえる(描く面と、クリックで当たる面を食い違わせない)
  if (findFaceSign(meshes) > 0) for (const m of meshes) m.flipFaces(false);
  faceSign = -1;
  for (const m of meshes) {
    if (!m.material) m.material = new BABYLON.StandardMaterial('scan', scene);
    const mat = m.material;
    mat.backFaceCulling = true;
    if (mat.diffuseTexture) {       // 写真つきなら、照明をかけずに写真の色のまま出す
      mat.emissiveTexture = mat.diffuseTexture;
      mat.disableLighting = true;
    } else {
      m.convertToFlatShadedMesh();   // 写真が無いときは、面ごとの陰で形が分かるようにする
      mat.diffuseColor = new BABYLON.Color3(0.8, 0.8, 0.82);
      mat.specularColor = new BABYLON.Color3(0, 0, 0);
    }
  }
  unit = Math.hypot(F.to_zup[0][0], F.to_zup[1][0], F.to_zup[2][0]);   // ファイルの1単位が何cmか
  const center = min.add(max).scale(0.5), size = max.subtract(min).length();
  cam = new BABYLON.ArcRotateCamera('cam', -Math.PI / 3, Math.PI / 3.2, size * 1.1, center, scene);
  cam.minZ = size / 2000; cam.maxZ = size * 20;
  cam.lowerRadiusLimit = size / 200; cam.upperRadiusLimit = size * 5;
  cam.wheelDeltaPercentage = 0.03;
  cam.panningSensibility = 2000 / size;
  cam.attachControl(canvas, true);
  // 回すためのドラッグでは置かない(動かさずに離したときだけ置く)
  scene.onPointerObservable.add(pi => {
    if (pi.type === BABYLON.PointerEventTypes.POINTERTAP && pi.event.button === 0) onTap(scene.pointerX, scene.pointerY);
  });
  document.getElementById('cut').oninput = setCut;
  engine.runRenderLoop(() => { stepPlay(); scene.render(); placeTag(); });
  document.getElementById('pos').oninput = ev => { stopPlay(); showAt(+ev.target.value); follow(1); };
  window.addEventListener('resize', () => engine.resize());
  info.textContent = '運ぶ人 ' + F.carriers + '人 / 左ドラッグで回す、右ドラッグで動かす、ホイールで寄る・引く。床をクリックするとピンが立つ' +
    ' (読み込み ' + ((Date.now() - t0) / 1000).toFixed(1) + '秒)';
  showPlaced();
}
// ---- 判定の結果を、同じ立体表示に重ねる ----
const C3 = hex => BABYLON.Color3.FromHexString(hex);
function plainMat(name, hex, lit) {
  const mat = new BABYLON.StandardMaterial(name, scene);
  if (lit) { mat.diffuseColor = C3(hex); mat.specularColor = new BABYLON.Color3(0, 0, 0); mat.twoSidedLighting = true; }
  else { mat.emissiveColor = C3(hex); mat.disableLighting = true; }
  mat.backFaceCulling = false;
  return mat;
}
// 頂点の並び(x, y, z, ...)と三角形から、後で頂点を差し替えられる部品を作る
function rawMesh(name, positions, indices, hex) {
  const m = new BABYLON.Mesh(name, scene), vd = new BABYLON.VertexData(), normals = [];
  BABYLON.VertexData.ComputeNormals(positions, indices, normals);
  vd.positions = positions; vd.indices = indices; vd.normals = normals;
  vd.applyToMesh(m, true);
  m.material = plainMat(name, hex, true);
  m.isPickable = false;
  m.alwaysSelectAsActiveMesh = true;   // 頂点を差し替えるので、古い範囲で描画を省かせない
  return m;
}
function setVerts(m, positions, indices) {
  const normals = [];
  BABYLON.VertexData.ComputeNormals(positions, indices, normals);
  m.updateVerticesData(BABYLON.VertexBuffer.PositionKind, positions);
  m.updateVerticesData(BABYLON.VertexBuffer.NormalKind, normals);
}
const V3 = (a, i) => new BABYLON.Vector3(a[i || 0], a[(i || 0) + 1], a[(i || 0) + 2]);
function lineColor(c) { return c === null ? '#8e44ad' : c < 0 ? '#c0392b' : c < 5 ? '#f1c40f' : '#27ae60'; }
function clearResult() {
  stopPlay();
  if (R) for (const m of R.meshes) m.dispose(false, true);
  R = null;
  document.getElementById('tag').style.display = 'none';
  document.getElementById('play').style.display = 'none';
}
function showResult(d) {
  clearResult();
  const meshes = [], fr = d.frames, w = d.bottleneck.frame;
  // 経路の線: 位置と位置の間を、両端の余裕の小さい方の色で塗る。壁の向こうでも見えるよう最後に描く
  for (let i = 0; i + 1 < fr.length; i++) {
    const a = fr[i].clearance, b = fr[i + 1].clearance;
    const t = BABYLON.MeshBuilder.CreateTube('seg', {path: [V3(fr[i].at), V3(fr[i + 1].at)], radius: 2.5 / unit, tessellation: 6}, scene);
    t.material = plainMat('seg', lineColor(a === null || b === null ? null : Math.min(a, b)), false);
    t.isPickable = false; t.renderingGroupId = 1;
    meshes.push(t);
  }
  const box = rawMesh('furniture', fr[w].box, d.box_tris, '#2c6fbb');
  const people = fr[w].people.map((v, k) => rawMesh('carrier' + k, v, d.people_tris, '#e07b39'));
  meshes.push(box, ...people);
  // 当たる所(一番余裕の小さい点)の印
  const ok = d.bottleneck.capacity_cm >= 0;
  const mark = BABYLON.MeshBuilder.CreateSphere('mark', {diameter: 12 / unit}, scene);
  mark.material = plainMat('mark', ok ? '#27ae60' : '#c0392b', false);
  mark.position = V3(d.bottleneck.contact);
  mark.isPickable = false; mark.renderingGroupId = 1;
  meshes.push(mark);
  R = {d, meshes, box, people, mark, t: w, playing: false};
  const tag = document.getElementById('tag');
  tag.textContent = d.reason.split('(')[0];
  tag.className = ok ? 'ok' : '';
  document.getElementById('headline').textContent = d.name + ' ' + d.size + ': ' + d.reason +
    ' / 家具だけなら' + d.furniture_only;
  const pos = document.getElementById('pos');
  pos.max = fr.length - 1;
  document.getElementById('play').style.display = 'block';
  goWorst();
}
// 2つの頂点の並びを、割合 u でつなぐ
function mix(a, b, u) {
  if (u <= 0) return a;
  if (u >= 1) return b;
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * u;
  return out;
}
// 経路上の位置 t(0〜位置の数-1。小数なら、前後の位置の間)の家具と人を表示する。
// 判定したのは整数の位置だけ。間は、見やすいよう形をつないでいるだけ
function showAt(t) {
  if (!R) return;
  const d = R.d, n = d.frames.length;
  t = Math.max(0, Math.min(n - 1, t));
  R.t = t;
  document.getElementById('pos').value = t;
  const i = Math.min(Math.floor(t), Math.max(n - 2, 0)), a = d.frames[i], b = d.frames[Math.min(i + 1, n - 1)];
  let u = t - i;
  // 片方に持てる姿勢が無いときは、つながずに近い方をそのまま出す
  if (a.box === null || b.box === null) u = u < 0.5 ? 0 : 1;
  const near = u < 0.5 ? a : b, has = near.box !== null;
  const c = !has ? null : (u > 0 && u < 1) ? a.clearance + (b.clearance - a.clearance) * u : near.clearance;
  const ng = has && c < 0;
  R.box.setEnabled(has);
  R.people.forEach(m => m.setEnabled(has));
  if (has) {
    setVerts(R.box, mix(a.box, b.box, u), d.box_tris);
    R.box.material.diffuseColor = C3(ng ? '#c0392b' : '#2c6fbb');
    R.people.forEach((m, k) => { setVerts(m, mix(a.people[k], b.people[k], u), d.people_tris); m.material.diffuseColor = C3(ng ? '#c0392b' : '#e07b39'); });
  }
  const sCm = a.s + (b.s - a.s) * (t - i);
  const where = '出発点から ' + (sCm / 100).toFixed(1) + 'm / ' + (d.length / 100).toFixed(1) + 'm: ';
  document.getElementById('posText').textContent = where + (!has ? '持てる姿勢が無い' :
    ng ? 'あと' + Math.max(1, Math.round(-c)) + 'cm足りない' : '余裕' + Math.round(c) + 'cm') +
    (Math.abs(t - d.bottleneck.frame) < 0.01 ? ' (一番狭い所)' : '');
}
// 一番狭い所を表示し、そこへ視点を寄せる。上の階の床で隠れないよう、その上を隠す
function goWorst() {
  if (!R) return;
  stopPlay();
  const d = R.d, w = d.bottleneck.frame;
  showAt(w);
  const s = document.getElementById('cut');
  const top = d.frames[w].at[1] + 240 / unit;
  if (top < bounds.max.y) { s.value = Math.ceil(s.max * (top - bounds.min.y) / (bounds.max.y - bounds.min.y)); setCut(); }
  cam.setTarget(V3(d.bottleneck.contact));
  cam.radius = 420 / unit;
  cam.beta = Math.min(cam.beta, 1.05);
  document.getElementById('view').scrollIntoView({behavior: 'smooth', block: 'center'});
}
function stopPlay() {
  if (R) R.playing = false;
  document.getElementById('playBtn').textContent = '▶ 運んでみる';
}
function togglePlay() {
  if (!R) return;
  if (R.playing) { stopPlay(); return; }
  if (R.t >= R.d.frames.length - 1) showAt(0);
  R.playing = true;
  document.getElementById('playBtn').textContent = '■ 止める';
}
const PLAY_SPEED = 60;   // 運んでいく速さ(cm/秒)
// 描くたびに呼ぶ: 再生中なら、経った時間のぶんだけ位置を進め、視点を家具に付いていかせる
function stepPlay() {
  if (!R || !R.playing) return;
  const d = R.d, n = d.frames.length;
  if (n < 2) { stopPlay(); return; }
  const i = Math.min(Math.floor(R.t), n - 2), ds = Math.max(1, d.frames[i + 1].s - d.frames[i].s);
  const dt = Math.min(engine.getDeltaTime(), 100) / 1000;   // 画面が止まっていた後に飛ばない
  const t = R.t + PLAY_SPEED * dt / ds;
  showAt(t);
  follow(Math.min(1, dt * 3));
  if (t >= n - 1) stopPlay();
}
// 視点の中心を、いま表示している位置へ寄せる(rate=1 ならすぐ、小さければ少しずつ)
function follow(rate) {
  const fr = R.d.frames, j = Math.min(Math.floor(R.t), Math.max(fr.length - 2, 0));
  const at = BABYLON.Vector3.Lerp(V3(fr[j].at), V3(fr[Math.min(j + 1, fr.length - 1)].at), R.t - j);
  at.y += 60 / unit;   // 床ではなく、家具の高さのあたりを見る
  cam.setTarget(BABYLON.Vector3.Lerp(cam.target, at, rate));
}
// 当たる所の印の上に、足りない量の札を出す(印が画面の外・カメラの後ろなら隠す)
function placeTag() {
  const tag = document.getElementById('tag');
  if (!R) return;
  const w = engine.getRenderWidth(), h = engine.getRenderHeight();
  const p = BABYLON.Vector3.Project(R.mark.position, BABYLON.Matrix.Identity(), scene.getTransformMatrix(), cam.viewport.toGlobal(w, h));
  const canvas = document.getElementById('view');
  if (p.z < 0 || p.z > 1 || p.x < 0 || p.x > w || p.y < 0 || p.y > h) { tag.style.display = 'none'; return; }
  tag.style.display = 'block';
  tag.style.left = (p.x * canvas.clientWidth / w) + 'px';
  tag.style.top = (p.y * canvas.clientHeight / h) + 'px';
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
      const b = el('button', '立体で見る');
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
    if (s.state === 'error') { viewStatus(name + ': 立体表示を作れませんでした: ' + s.message); return; }
    viewStatus('');
    showResult(s.scene);
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
  clearResult();
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
  startView();
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

        def _file(self, folder, url_path, types):
            """folder の直下にある、決めた種類のファイルだけを配る。"""
            name = plain_name(unquote(url_path))
            f = Path(folder) / name if name else None
            ctype = types.get(f.suffix.lower()) if f else None
            if ctype is None or not f.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(f.stat().st_size))
            self.end_headers()
            with open(f, "rb") as src:
                shutil.copyfileobj(src, self.wfile)

        def _json(self, obj):
            self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path.startswith("/vendor/"):
                self._file(fetch_viewer.VENDOR_DIR, u.path, VENDOR_TYPES)
            elif u.path.startswith("/scan/"):
                self._file(app.view["dir"], u.path, VIEW_TYPES)
            elif u.path == "/floor":
                self._json(app.floor_json())
            elif u.path == "/status":
                job = parse_qs(u.query).get("job", [""])[0]
                self._json(app.jobs.get(job, {"state": "error", "message": "不明なジョブ"}))
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
        view = view_of_mesh(mesh)
    elif args.mesh:
        import scan_ingest as SI
        mesh, info = SI.load_scan_zup(args.mesh)
        view = view_of_scan(args.mesh, info["to_zup"])
        if view is None:
            print("[picker] OBJ ではないので、写真なしの形だけを表示します")
            view = view_of_mesh(mesh)
    else:
        parser.error("mesh を指定する(または --route)")
    if fetch_viewer.missing():
        print("[picker] 立体表示の部品がありません。先に python fetch_viewer.py を実行してください")
    app = App(mesh, view, args.carriers, seed=args.seed, items_text=items_text)
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
