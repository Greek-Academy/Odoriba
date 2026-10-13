"""
展示モニター用の集客画面(自動ループ)を書き出す。

ブースのモニターに全画面で流しておき、通りがかりの人の足を止めるための画面。
操作はいらない。シーンが順に切り替わり、最後に「このブースで試せます」へ誘導する。

アニメーションは作り物ではなく、results/lstair_result.json に保存した
実際の計算結果(家具だけで通った経路と、2人で運んで詰まるまでの経路)を
上から見た図にしたもの。

    cd phase2
    python attract.py                 # ../demo/attract.html を書き出す

kiosk.py を起動していれば http://localhost:8000/demo/attract.html で開ける
(ファイルを直接ダブルクリックしても動く)。F11 で全画面にする。
"""

import argparse
import json
from pathlib import Path

import numpy as np

import geometry3d as g3
import phase2_lstair as L

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results" / "lstair_result.json"
OUT = ROOT.parent / "demo" / "attract.html"

# 幅80cmのふつうの階段での判定(kiosk.pyのプリセットを掃引した値)
EXAMPLES = [
    ("本棚", "180×40×30", 18, -14),
    ("冷蔵庫", "180×65×65", 8, -28),
    ("タンス", "200×50×65", 8, -34),
    ("小さめの収納棚", "80×45×40", 18, 1),
]


def _hull(pts):
    """2次元点群の凸包(反時計回り)。Andrewの単調連鎖法。"""
    pts = sorted(map(tuple, pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _frame(pos, quat, with_carriers):
    """1姿勢ぶんの、上から見た家具の輪郭と運搬者の位置。"""
    R = g3.rotmat_from_quat(quat)
    hl, hw, hh = L.FURN_L / 2, L.FURN_W / 2, L.FURN_H / 2
    corners = np.array([[sx * hl, sy * hw, sz * hh]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = corners @ R.T + np.asarray(pos)
    fr = {"p": [[round(float(x), 1), round(float(y), 1)] for x, y in _hull(world[:, :2])]}
    if with_carriers:
        # 運搬者は長軸の両端から腕の長さだけ外に立つ(place_humans_lstairと同じ置き方)。
        # 詰まる姿勢では足元に床が無い(=立てない)こともあるので、床の有無は見ずに描く
        fr["c"] = []
        for sign in (1.0, -1.0):
            g = np.asarray(pos) + R @ np.array([sign * (hl + L.p.CARRY_ARM), 0.0, 0.0])
            fr["c"].append([round(float(g[0]), 1), round(float(g[1]), 1)])
    return fr


def _densify(path, step_cm=6.0):
    """経路の姿勢列を、動きが滑らかに見える細かさまで補間する。"""
    out = []
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        # 回転も動きとして数える(1度 ≒ 1cm)
        ang = np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(a[3:], b[3:]))))))
        n = max(1, int(np.ceil(max(np.linalg.norm(b[:3] - a[:3]), ang) / step_cm)))
        for i in range(n):
            pos, q = g3.interpolate_se3(a[:3], a[3:], b[:3], b[3:], i / n)
            out.append((pos, q))
    out.append((np.asarray(path[-1][:3], float), np.asarray(path[-1][3:], float)))
    return out


def build_data():
    res = json.loads(RESULT.read_text(encoding="utf-8"))
    st, fu = res["meta"]["stair"], res["meta"]["furniture"]
    L.configure(width=st["width"], rise=st["rise"], tread=st["tread"],
                n_steps1=st["n_steps1"], n_steps2=st["n_steps2"])
    L.FURN_L, L.FURN_W, L.FURN_H = fu["L"], fu["W"], fu["H"]

    alone = [_frame(p, q, False) for p, q in _densify(res["box_only"]["path"])]
    # 2人: 進めたところまでの経路のあと、一番狭い場所での最良の姿勢で止まる
    bp = res["with_carriers"]["best_effort_path"]
    pose_pos, pose_q = res["bottleneck"]["pose"]
    stuck_path = bp + [list(pose_pos) + list(pose_q)]
    carried = [_frame(p, q, True) for p, q in _densify(stuck_path)]

    return {
        "stair": {"w": L.STAIR_WIDTH, "fl1y0": L.FL1_Y0, "fl1y1": L.FL1_Y1,
                  "landy1": L.LAND_Y1, "fl2x0": L.FL2_X0, "fl2x1": L.FL2_X1,
                  "topx1": L.TOP_X1, "tread": L.TREAD,
                  "n1": L.N_STEPS1, "n2": L.N_STEPS2},
        "carrier_r": res["meta"]["carrier"]["r"],
        "alone": alone,
        "carried": carried,
        "margin_alone": round(res["bottleneck_furniture_only"]["capacity_cm"]),
        "short_carried": round(-res["bottleneck"]["capacity_cm"]),
        "max_len_carried": round(res["max_length_with_carriers"]["max_l"]),
        "furn": f'{fu["L"]:.0f}×{fu["W"]:.0f}×{fu["H"]:.0f}',
        "stair_w": round(st["width"]),
    }


# 上から見た階段と家具を描くJS(poster.pyでも使う)。Dに build_data() の中身が入っている前提
DRAW_JS = r"""function setupCanvas(cv) {
  const r = window.devicePixelRatio || 1, b = cv.getBoundingClientRect();
  cv.width = b.width * r; cv.height = b.height * r;
  const ctx = cv.getContext("2d"); ctx.setTransform(r, 0, 0, r, 0, 0);
  // 描く範囲: 下の廊下の途中〜上の廊下の途中(cm)
  const S = D.stair, x0 = -60, x1 = S.fl2x1 + 160, y0 = 0, y1 = S.landy1 + 60;
  const k = Math.min(b.width / (x1 - x0), b.height / (y1 - y0));
  const ox = (b.width - (x1 - x0) * k) / 2, oy = (b.height - (y1 - y0) * k) / 2;
  return {ctx, w: b.width, h: b.height, k,
          X: x => ox + (x - x0) * k, Y: y => b.height - oy - (y - y0) * k};
}

function drawStair(v) {
  const {ctx, X, Y, k} = v, S = D.stair;
  ctx.clearRect(0, 0, v.w, v.h);
  // 通れる床
  ctx.fillStyle = "#e9e3d5";
  ctx.fillRect(X(0), Y(S.landy1), S.w * k, (S.landy1 - 0) * k);
  ctx.fillRect(X(0), Y(S.landy1), (S.topx1) * k, S.w * k);
  // 段(下側=縦、上側=横)と踊り場
  ctx.fillStyle = "#d6cdb8";
  ctx.fillRect(X(0), Y(S.fl1y1), S.w * k, (S.fl1y1 - S.fl1y0) * k);
  ctx.fillRect(X(S.fl2x0), Y(S.landy1), (S.fl2x1 - S.fl2x0) * k, S.w * k);
  ctx.fillStyle = "#efd9a0";
  ctx.fillRect(X(0), Y(S.landy1), S.w * k, S.w * k);
  ctx.strokeStyle = "#b9ae96"; ctx.lineWidth = 1;
  for (let i = 1; i < S.n1; i++) { const y = S.fl1y0 + i * S.tread;
    ctx.beginPath(); ctx.moveTo(X(0), Y(y)); ctx.lineTo(X(S.w), Y(y)); ctx.stroke(); }
  for (let i = 1; i < S.n2; i++) { const x = S.fl2x0 + i * S.tread;
    ctx.beginPath(); ctx.moveTo(X(x), Y(S.fl1y1)); ctx.lineTo(X(x), Y(S.landy1)); ctx.stroke(); }
  // 壁
  ctx.strokeStyle = "#3d3b37"; ctx.lineWidth = 3; ctx.beginPath();
  ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(0), Y(S.landy1)); ctx.lineTo(X(S.topx1), Y(S.landy1));
  ctx.moveTo(X(S.w), Y(0)); ctx.lineTo(X(S.w), Y(S.fl1y1)); ctx.lineTo(X(S.topx1), Y(S.fl1y1));
  ctx.stroke();
  ctx.fillStyle = "#8a7350"; ctx.font = `${Math.max(12, 18 * k)}px "Yu Gothic UI"`;
  ctx.textAlign = "center"; ctx.fillText("踊り場", X(S.w / 2), Y(S.fl1y1 + S.w / 2) + 6);
  ctx.fillText("上の階へ →", X(S.fl2x1 + 70), Y(S.fl1y1 + S.w / 2) + 6);
}

function drawFrame(v, fr, color, stuck) {
  const {ctx, X, Y, k} = v;
  ctx.fillStyle = color; ctx.strokeStyle = "#2b2a28"; ctx.lineWidth = 1.5;
  ctx.beginPath(); fr.p.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)));
  ctx.closePath(); ctx.fill(); ctx.stroke();
  (fr.c || []).forEach(([x, y]) => {
    ctx.fillStyle = stuck ? "#b85042" : "#e8b07e";
    ctx.beginPath(); ctx.arc(X(x), Y(y), D.carrier_r * k, 0, 7); ctx.fill(); ctx.stroke();
  });
}
"""


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>Odoriba</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;overflow:hidden;cursor:none}
body{font-family:"Yu Gothic UI","Meiryo",sans-serif;background:#f4f1ea;color:#2b2a28}
.mincho{font-family:"Yu Mincho","YuMincho","Hiragino Mincho ProN",serif;font-weight:600}
.scene{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;
  align-items:center;text-align:center;padding:4vh 6vw 10vh;opacity:0;transition:opacity .9s}
.scene.on{opacity:1}
.dark{background:#1f3a2e;color:#f4f1ea}
.h1{font-size:6.2vw;line-height:1.35;letter-spacing:.06em}
.h2{font-size:4.4vw;line-height:1.4;letter-spacing:.05em}
.sub{font-size:2.3vw;line-height:1.7;margin-top:3vh;color:#cfc6b2}
.light .sub{color:#5d574c}
.big{font-size:13vw;line-height:1;color:#d9a679;margin:2vh 0}
.accent{color:#d9a679}
.ng{color:#7a2e2e}.ok{color:#1f3a2e}
.brand{position:absolute;left:3vw;bottom:3vh;font-size:1.8vw;letter-spacing:.12em;color:#8a7350;z-index:5}
.dots{position:absolute;right:3vw;bottom:3.6vh;display:flex;gap:.8vw;z-index:5}
.dots i{width:1vw;height:1vw;border-radius:50%;background:#cfc6b2;transition:.4s}
.dots i.on{background:#8a7350}
#s3{justify-content:flex-start;padding-top:4vh;padding-bottom:13vh}
.panels{display:flex;gap:3vw;width:100%;flex:1;margin-top:2vh}
.panel{flex:1;display:flex;flex-direction:column;align-items:center}
.panel h3{font-size:2.6vw;margin-bottom:1vh}
.panel canvas{width:100%;flex:1;min-height:0}
.verdict{font-size:3vw;height:6vh;opacity:0;transition:opacity .5s;margin-top:1vh}
.verdict.on{opacity:1}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:2.4vh 2.4vw;width:88%;margin-top:4vh}
.card{background:#fbfaf6;border:1px solid #d9d2c3;border-radius:6px;padding:2.2vh 2vw;text-align:left}
.card b{font-size:2.5vw}.card small{font-size:1.6vw;color:#7a7468;margin-left:.6vw}
.card p{font-size:2.1vw;margin-top:1vh;line-height:1.5}
.steps{display:flex;gap:2.5vw;margin-top:5vh}
.step{background:#2b4a3c;border-radius:8px;padding:3vh 2vw;width:22vw;font-size:2.2vw;line-height:1.5}
.step em{display:block;font-style:normal;font-size:3.6vw;color:#d9a679;margin-bottom:1vh}
.arrow{font-size:6vw;color:#d9a679;margin-top:4vh;animation:bob 1.2s ease-in-out infinite}
@keyframes bob{50%{transform:translateY(1.5vh)}}
</style></head><body>

<section class="scene dark" id="s0">
  <div class="h1 mincho">その家具、<br>部屋に置けても、<br><span class="accent">そこまで運べますか？</span></div>
</section>

<section class="scene dark" id="s1">
  <div class="h2 mincho">入らなかった家具は、<br>窓からクレーンで吊り上げる。</div>
  <div class="big mincho">2万〜5万円</div>
  <div class="sub">しかも分かるのは、いつも<b>買った後</b>。</div>
</section>

<section class="scene light" id="s2">
  <div class="h2 mincho">いまの計算は、<br>家具が<span class="accent" style="color:#b85042">宙に浮いて</span>勝手に動く前提。</div>
  <div class="sub" style="font-size:2.8vw">でも実際は、<b>人が持って運ぶ</b>。<br>人の体も、通り道をふさぐ。</div>
</section>

<section class="scene light" id="s3">
  <div class="h2 mincho" style="font-size:3.2vw">同じタンス（__FURN__cm）・同じ階段（幅__SW__cm）を、上から見ると</div>
  <div class="panels">
    <div class="panel"><h3 class="mincho ok">家具だけ</h3><canvas id="cA"></canvas>
      <div class="verdict mincho ok" id="vA">○ 通る（余裕 __MA__cm）</div></div>
    <div class="panel"><h3 class="mincho ng">2人で持って運ぶ</h3><canvas id="cB"></canvas>
      <div class="verdict mincho ng" id="vB">✕ 踊り場で あと __SC__cm</div></div>
  </div>
</section>

<section class="scene dark" id="s4">
  <div class="h1 mincho" style="font-size:5.2vw">人が運ぶと考えるだけで、<br><span class="accent">答えが逆になる。</span></div>
  <div class="sub" style="font-size:2.6vw">この階段なら、2人で運べるタンスは<b style="color:#f4f1ea">長さ __ML__cm まで</b>。<br>「どこで・あと何cm」まで計算します。</div>
</section>

<section class="scene light" id="s5">
  <div class="h2 mincho" style="font-size:3.4vw">ふつうの家の階段（幅80cm）だと…</div>
  <div class="cards">__CARDS__</div>
</section>

<section class="scene dark" id="s6">
  <div class="h1 mincho" style="font-size:5.4vw">このブースで、<br><span class="accent">試せます。</span></div>
  <div class="steps">
    <div class="step"><em>①</em>階段をえらぶ</div>
    <div class="step"><em>②</em>家具をえらぶ</div>
    <div class="step"><em>③</em>すぐ結果が出る</div>
  </div>
  <div class="sub" style="font-size:2.6vw">あなたの家の階段の寸法でも、どうぞ。</div>
  <div class="arrow">▼</div>
</section>

<div class="brand mincho">Odoriba（おどりば）｜ 運び込めるかを、運ぶ人ごと計算する</div>
<div class="dots" id="dots"></div>

<script>
const D = __DATA__;
// [シーンid, 表示ミリ秒]。s3はアニメーションの長さに合わせて延ばす
const SCENES = [["s0", 6000], ["s1", 7000], ["s2", 7000], ["s3", 0],
                ["s4", 7500], ["s5", 9000], ["s6", 8000]];
const dots = document.getElementById("dots");
SCENES.forEach(() => dots.appendChild(document.createElement("i")));

__DRAW_JS__
// 2枚を同時に動かし、2人の側は経路の終わり(一番狭い場所)で止めて赤くする
function playS3(done) {
  const a = setupCanvas(document.getElementById("cA")), b = setupCanvas(document.getElementById("cB"));
  const vA = document.getElementById("vA"), vB = document.getElementById("vB");
  vA.classList.remove("on"); vB.classList.remove("on");
  const FPS = 1000 / 30, nA = D.alone.length, nB = D.carried.length;
  let i = 0, blink = 0;
  const timer = setInterval(() => {
    const iA = Math.min(i, nA - 1), iB = Math.min(i, nB - 1), stuck = i >= nB - 1;
    drawStair(a); drawFrame(a, D.alone[iA], "#2f6b52", false);
    drawStair(b);
    const red = stuck && Math.floor(blink / 8) % 2 === 0;
    drawFrame(b, D.carried[iB], red ? "#7a2e2e" : "#2f6b52", stuck);
    if (i === nA - 1) vA.classList.add("on");
    if (stuck) {
      vB.classList.add("on"); blink++;
      const S = D.stair, c = b.ctx;
      c.fillStyle = "#7a2e2e"; c.font = `bold ${Math.max(18, 30 * b.k)}px "Yu Gothic UI"`;
      c.textAlign = "left";
      c.fillText("← どう向けても、壁や段にぶつかる", b.X(S.w + 40), b.Y(S.landy1 + 30));
    }
    i++;
    if (i > Math.max(nA, nB) + 75) { clearInterval(timer); done(); }
  }, FPS);
}

let cur = 0;
function show(n) {
  document.querySelectorAll(".scene").forEach(s => s.classList.remove("on"));
  [...dots.children].forEach((d, j) => d.classList.toggle("on", j === n));
  const [id, ms] = SCENES[n];
  document.getElementById(id).classList.add("on");
  const next = () => show((n + 1) % SCENES.length);
  if (id === "s3") setTimeout(() => playS3(next), 700); else setTimeout(next, ms);
}
// URLの#数字で開始シーンを指定できる(確認用。例: attract.html#3)
show(parseInt(location.hash.slice(1)) || 0);
</script></body></html>"""


def render(data):
    cards = []
    for name, dims, alone, carried in EXAMPLES:
        tail = (f'<span class="ok">○ ぎりぎり通る（余裕{carried}cm）</span>' if carried >= 0
                else f'<span class="ng">✕ 踊り場で あと{-carried}cm</span>')
        cards.append(f'<div class="card"><b class="mincho">{name}</b><small>{dims}cm</small>'
                     f'<p>家具だけ：<span class="ok">○ 通る（余裕{alone}cm）</span><br>'
                     f'2人で運ぶ：{tail}</p></div>')
    return (PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__CARDS__", "".join(cards)).replace("__DRAW_JS__", DRAW_JS)
            .replace("__FURN__", data["furn"]).replace("__SW__", str(data["stair_w"]))
            .replace("__MA__", str(data["margin_alone"]))
            .replace("__SC__", str(data["short_carried"]))
            .replace("__ML__", str(data["max_len_carried"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data = build_data()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    print(f"書き出し: {out}  (家具だけ {len(data['alone'])}コマ / 2人 {len(data['carried'])}コマ)")


if __name__ == "__main__":
    main()
