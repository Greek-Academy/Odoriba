"""
展示モニター用の「一目で分かる」1枚画面を書き出す。

attract.py(自動で切り替わるループ)とは別に、通りがかりの人が数秒見ただけで
「何の作品か」「何が新しいか」「ここで試せる」が分かる静止画面。
図は attract.py と同じ実際の計算結果(results/lstair_result.json)から描く。

    cd phase2
    python poster.py              # ../demo/poster.html を書き出す
    python poster.py --png        # あわせて ../demo/poster.png (1920x1080) も書き出す

--png はインストール済みの Chrome / Edge をヘッドレスで使って撮影する。
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import attract as A

OUT = A.ROOT.parent / "demo" / "poster.html"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "chrome", "msedge", "google-chrome",
]

PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>Odoriba</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;overflow:hidden}
body{font-family:"Yu Gothic UI","Meiryo",sans-serif;background:#f4f1ea;color:#2b2a28;
  display:grid;grid-template-rows:auto minmax(0,1fr) auto;cursor:none}
.mincho{font-family:"Yu Mincho","YuMincho","Hiragino Mincho ProN",serif;font-weight:600}
header{background:#1f3a2e;color:#f4f1ea;padding:2.6vh 4vw 2.4vh;display:flex;align-items:flex-end;
  justify-content:space-between}
header h1{font-size:3.7vw;line-height:1.3;letter-spacing:.05em}
header h1 span{color:#d9a679}
header .name{text-align:right;font-size:1.5vw;color:#cfc6b2;letter-spacing:.12em;line-height:1.6}
header .name b{display:block;font-size:2.6vw;color:#f4f1ea;letter-spacing:.08em}
main{display:grid;grid-template-columns:1.45fr 1fr;gap:3vw;padding:2.4vh 4vw 2.4vh;min-height:0;overflow:hidden}
.viz{display:flex;flex-direction:column;min-height:0}
.viz .cap{font-size:1.4vw;color:#5d574c;margin-bottom:1vh}
.panels{display:flex;gap:1.6vw;flex:1;min-height:0}
.panel{flex:1;display:flex;flex-direction:column;background:#fbfaf6;border:1px solid #d9d2c3;
  border-radius:6px;padding:1.4vh 1vw;min-height:0}
.panel h3{font-size:1.7vw}
.panel canvas{flex:1;width:100%;min-height:0}
.verdict{font-size:2.1vw;line-height:1.3;white-space:nowrap}
.ok{color:#1f3a2e}.ng{color:#7a2e2e}
.points{display:flex;flex-direction:column;justify-content:center;gap:3vh}
.pt{display:flex;gap:1.2vw;align-items:flex-start}
.pt i{flex:none;width:2.8vw;height:2.8vw;border-radius:50%;background:#8a7350;color:#fbfaf6;
  font-style:normal;font-size:1.5vw;display:flex;align-items:center;justify-content:center}
.pt:nth-child(2) i{background:#b85042}.pt:nth-child(3) i{background:#1f3a2e}
.pt h4{font-size:1.65vw;line-height:1.4;white-space:nowrap;letter-spacing:0}
.pt p{font-size:1.2vw;line-height:1.55;color:#5d574c;margin-top:.5vh}
.pt em{font-style:normal;color:#b85042;font-weight:bold}
footer{background:#b85042;color:#fbfaf6;padding:2vh 4vw;white-space:nowrap;display:flex;align-items:center;
  justify-content:space-between}
footer .go{font-size:2.3vw;letter-spacing:.06em}
footer .how{font-size:1.2vw;opacity:.95}
</style></head><body>

<header>
  <h1 class="mincho">その家具、部屋に置けても、<br><span>そこまで運べますか？</span></h1>
  <div class="name mincho"><b>Odoriba（おどりば）</b>家具が運び込めるかを、<br>運ぶ人ごと計算するアプリ</div>
</header>

<main>
  <section class="viz">
    <div class="cap">同じタンス（__FURN__cm）・同じ階段（幅__SW__cm）を上から見ると</div>
    <div class="panels">
      <div class="panel"><h3 class="mincho ok">家具だけ</h3><canvas id="cA"></canvas>
        <div class="verdict mincho ok">○ 通る<br><small style="font-size:.7em">一番狭い所で 余裕__MA__cm</small></div></div>
      <div class="panel"><h3 class="mincho ng">2人で持って運ぶ</h3><canvas id="cB"></canvas>
        <div class="verdict mincho ng">✕ 通らない<br><small style="font-size:.7em">踊り場で あと__SC__cm</small></div></div>
    </div>
  </section>

  <section class="points">
    <div class="pt"><i class="mincho">1</i><div><h4 class="mincho">入らない家具は、クレーンで<em>2万〜5万円</em></h4>
      <p>窓から吊り上げる専門の業者さんがいるほど。しかも分かるのは、いつも買った後。</p></div></div>
    <div class="pt"><i class="mincho">2</i><div><h4 class="mincho">運ぶ人の体まで、計算に入れる</h4>
      <p>いまの計算は「家具が宙に浮いて動く」前提。実際は人が運ぶので、人の体も通り道をふさぐ。</p></div></div>
    <div class="pt"><i class="mincho">3</i><div><h4 class="mincho">「どこで・あと何cm」まで答える</h4>
      <p>通る／通らないだけでなく、詰まる場所と足りない長さを数字で。買う前に分かる。</p></div></div>
  </section>
</main>

<footer>
  <div class="go mincho">このブースで、あなたの家の階段でも試せます →</div>
  <div class="how">階段と家具をえらぶだけ ・ 試作品による目安です</div>
</footer>

<script>
const D = __DATA__;
__DRAW_JS__
// 通った跡を薄く重ね、家具だけは踊り場を回っている姿勢、2人は詰まった姿勢を描く
function trail(v, frames, every, color) {
  v.ctx.globalAlpha = 0.16;
  frames.forEach((fr, i) => { if (i % every === 0) drawFrame(v, {p: fr.p}, color, false); });
  v.ctx.globalAlpha = 1;
}
function render() {
  const a = setupCanvas(document.getElementById("cA")), b = setupCanvas(document.getElementById("cB"));
  const S = D.stair;
  drawStair(a); trail(a, D.alone, 8, "#2f6b52");
  drawFrame(a, D.alone[D.alone.length - 1], "#2f6b52", false);
  // 踊り場で向きを変えている途中の姿勢(中心が踊り場に一番近いコマ)を濃く描く
  const mid = D.alone.reduce((best, fr, i) => {
    const cx = fr.p.reduce((s, q) => s + q[0], 0) / fr.p.length;
    const cy = fr.p.reduce((s, q) => s + q[1], 0) / fr.p.length;
    const d = Math.hypot(cx - S.w / 2, cy - (S.fl1y1 + S.w / 2));
    return d < best[0] ? [d, i] : best;
  }, [1e9, 0])[1];
  drawFrame(a, D.alone[mid], "#2f6b52", false);
  drawStair(b); trail(b, D.carried, 6, "#2f6b52");
  drawFrame(b, D.carried[D.carried.length - 1], "#7a2e2e", true);
  const c = b.ctx;
  c.fillStyle = "#7a2e2e"; c.font = `bold ${Math.max(16, 34 * b.k)}px "Yu Gothic UI"`;
  c.textAlign = "left";
  c.fillText("← ここで詰まる", b.X(S.w + 30), b.Y(S.landy1 + 25));
}
window.addEventListener("load", render);
window.addEventListener("resize", render);
</script></body></html>"""


def render(data):
    return (PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__DRAW_JS__", A.DRAW_JS)
            .replace("__FURN__", data["furn"]).replace("__SW__", str(data["stair_w"]))
            .replace("__MA__", str(data["margin_alone"]))
            .replace("__SC__", str(data["short_carried"])))


def screenshot(html_path, png_path):
    """Chrome/Edgeのヘッドレスで1920x1080のPNGに撮る。"""
    exe = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if exe is None:
        raise SystemExit("Chrome / Edge が見つからないため PNG を書き出せません")
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--window-size=1920,1080", "--virtual-time-budget=2000",
                    f"--screenshot={png_path}", html_path.resolve().as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--png", action="store_true", help="PNG(1920x1080)も書き出す")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(A.build_data()), encoding="utf-8")
    print(f"書き出し: {out}")
    if args.png:
        png = out.with_suffix(".png")
        screenshot(out, png)
        print(f"書き出し: {png}")


if __name__ == "__main__":
    main()
