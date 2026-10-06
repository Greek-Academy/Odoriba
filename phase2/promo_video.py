"""
展示モニター用の約60秒の紹介動画(MP4)を書き出す。

音なし・文字だけで伝わる構成。流れは「問題 → 新しい考え方 → ①スマホ(LiDAR)で
スキャン → ②寸法を読み取る → ③運ぶ人ごと計算 → ブースで試せる」。
スキャンの場面は実際に撮った階段の点群、寸法は measure_stairs の計測値、
比較アニメーションは attract.py と同じ計算結果(results/lstair_result.json)。

    cd phase2
    python promo_video.py                 # ../demo/odoriba_60s.mp4
    python promo_video.py --still 30      # 30秒目の1コマだけPNGに書き出す(確認用)

必要: pillow, imageio-ffmpeg (pip install imageio-ffmpeg)
"""

import argparse
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import attract as A

W, H, FPS = 1920, 1080, 30
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "odoriba_60s.mp4"
PHOTOS = ROOT / "docs" / "assets" / "syasin"
# iPhone(LiDAR)+Scaniverseで撮った実際の階段。git管理外の置き場にある
SCAN_OBJ = Path(r"C:\dev\alpha-project\Scaniverse 2026-09-19 113017 stair1"
                r"\Scaniverse 2026-09-19 113017\Scaniverse_2026_09_19_113017.obj")

# 配色(poster.png / kiosk と同じ)
BG = (244, 241, 234)
DARK = (31, 58, 46)
TERRA = (184, 80, 66)
BRASS = (138, 115, 80)
GOLD = (217, 166, 121)
INK = (43, 42, 40)
SUB = (93, 87, 76)
PALE = (207, 198, 178)
OK = (31, 58, 46)
NG = (122, 46, 46)
FURN = (47, 107, 82)
CARRIER = (232, 176, 126)
CARRIER_NG = (184, 80, 66)

FONT_DIR = Path("C:/Windows/Fonts")
MINCHO = FONT_DIR / "yumindb.ttf"
GOTHIC = FONT_DIR / "YuGothM.ttc"
GOTHIC_B = FONT_DIR / "YuGothB.ttc"


@lru_cache(maxsize=None)
def font(path, size):
    return ImageFont.truetype(str(path), size)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    """0..1 をなめらかに(始めと終わりがゆっくり)。"""
    x = clamp(x)
    return x * x * (3 - 2 * x)


def appear(t, start, dur=0.6):
    return ease((t - start) / dur)


def mix(c, bg, a):
    """色cを背景bgに不透明度aで重ねた色。"""
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


def text_lines(d, lines, cx, y, fnt, bg, alpha=1.0, gap=1.45, anchor="center", dy=0):
    """lines: [[(文字, 色), ...], ...]。行ごとに中央(または左)揃えで描く。
    alphaで背景から浮かび上がらせ、dyで少し下から上がってくる動きをつける。"""
    size = fnt.size
    for i, segs in enumerate(lines):
        width = sum(fnt.getlength(s) for s, _ in segs)
        x = cx - width / 2 if anchor == "center" else cx
        yy = y + i * size * gap + dy * (1 - alpha)
        for s, col in segs:
            d.text((x, yy), s, font=fnt, fill=mix(col, bg, alpha))
            x += fnt.getlength(s)
    return y + len(lines) * size * gap


# ---------------- 上から見た階段(attract.pyのdrawStair/drawFrameのPIL版) ----------------

class View:
    """cm座標 -> 画像ピクセル座標。y は上向き。"""

    def __init__(self, S, w, h, x0=-60, x1=None, y0=0, y1=None):
        x1 = S["fl2x1"] + 300 if x1 is None else x1
        y1 = S["landy1"] + 70 if y1 is None else y1
        self.k = min(w / (x1 - x0), h / (y1 - y0))
        self.ox = (w - (x1 - x0) * self.k) / 2 - x0 * self.k
        self.oy = h - (h - (y1 - y0) * self.k) / 2 + y0 * self.k

    def __call__(self, x, y):
        return (self.ox + x * self.k, self.oy - y * self.k)


def draw_stair(d, v, S, ss):
    def rect(x0, y0, x1, y1, fill):
        a, b = v(x0, y1), v(x1, y0)
        d.rectangle([a, b], fill=fill)

    rect(0, 0, S["w"], S["landy1"], (233, 227, 213))
    rect(0, S["fl1y1"], S["topx1"], S["landy1"], (233, 227, 213))
    rect(0, S["fl1y0"], S["w"], S["fl1y1"], (214, 205, 184))
    rect(S["fl2x0"], S["fl1y1"], S["fl2x1"], S["landy1"], (214, 205, 184))
    rect(0, S["fl1y1"], S["w"], S["landy1"], (239, 217, 160))
    for i in range(1, S["n1"]):
        y = S["fl1y0"] + i * S["tread"]
        d.line([v(0, y), v(S["w"], y)], fill=(185, 174, 150), width=max(1, ss))
    for i in range(1, S["n2"]):
        x = S["fl2x0"] + i * S["tread"]
        d.line([v(x, S["fl1y1"]), v(x, S["landy1"])], fill=(185, 174, 150), width=max(1, ss))
    wall = dict(fill=(61, 59, 55), width=4 * ss)
    d.line([v(0, 0), v(0, S["landy1"]), v(S["topx1"], S["landy1"])], **wall, joint="curve")
    d.line([v(S["w"], 0), v(S["w"], S["fl1y1"]), v(S["topx1"], S["fl1y1"])], **wall, joint="curve")
    f = font(GOTHIC, int(15 * ss * v.k * 1.4))
    c = v(S["w"] / 2, S["fl1y1"] + S["w"] / 2)
    d.text(c, "踊り場", font=f, fill=BRASS, anchor="mm")
    c = v(S["fl2x1"] + 80, S["fl1y1"] + S["w"] / 2)
    d.text(c, "上の階へ →", font=f, fill=BRASS, anchor="mm")


def draw_furn(d, v, fr, color, ss, carriers_color=None, r_cm=20.0):
    d.polygon([v(x, y) for x, y in fr["p"]], fill=color, outline=INK, width=2 * ss)
    for x, y in fr.get("c", []):
        cx, cy = v(x, y)
        r = r_cm * v.k
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=carriers_color or CARRIER,
                  outline=INK, width=2 * ss)


def stair_panel(data, w, h, frames=(), ss=2, label=None, label_pos=None):
    """frames: [(コマ, 色, 運搬者の色 or None)] を重ねた上面図の画像(RGBA)。"""
    img = Image.new("RGB", (w * ss, h * ss), (251, 250, 246))
    d = ImageDraw.Draw(img)
    v = View(data["stair"], w * ss, h * ss)
    draw_stair(d, v, data["stair"], ss)
    for fr, col, ccol in frames:
        draw_furn(d, v, fr, col, ss, ccol, data["carrier_r"])
    if label:
        f = font(GOTHIC_B, int(34 * ss))
        d.text(v(*label_pos), label, font=f, fill=NG, anchor="lm")
    return img.resize((w, h), Image.LANCZOS)


# ---------------- 場面 ----------------
# 各場面は (長さ秒, 背景色, 描画関数) 。描画関数は (img, draw, 場面内の秒) を受け取る。

def s_hook(img, d, t):
    f = font(MINCHO, 104)
    text_lines(d, [[("その家具、", BG)]], W / 2, 250, f, DARK, appear(t, 0.2), dy=30)
    text_lines(d, [[("部屋に置けても、", BG)]], W / 2, 400, f, DARK, appear(t, 0.9), dy=30)
    text_lines(d, [[("そこまで運べますか？", GOLD)]], W / 2, 560, f, DARK, appear(t, 1.8), dy=30)


def s_crane(img, d, t):
    # 家と窓
    ground = 900
    d.line([(80, ground), (900, ground)], fill=SUB, width=4)
    d.rectangle([480, 420, 860, ground], fill=(233, 227, 213), outline=SUB, width=4)
    d.polygon([(450, 420), (670, 280), (890, 420)], fill=(214, 205, 184), outline=SUB)
    d.rectangle([560, 480, 700, 600], fill=(251, 250, 246), outline=SUB, width=4)  # 2階の窓
    d.rectangle([560, 690, 700, 810], fill=(251, 250, 246), outline=SUB, width=4)
    # クレーン(トラック+ブーム)
    d.rectangle([110, 820, 340, ground - 10], fill=BRASS)
    d.ellipse([130, 860, 180, 910], fill=INK)
    d.ellipse([270, 860, 320, 910], fill=INK)
    tip = (630, 300)
    d.line([(220, 830), tip], fill=GOLD, width=14)
    # 吊られた家具が窓の高さまで上がる
    rise = ease(t / 5.0)
    by = 790 - rise * 300
    d.line([tip, (630, by - 40)], fill=INK, width=3)
    d.rectangle([570, by - 40, 690, by + 40], fill=FURN, outline=INK, width=3)
    d.text((780, 250), "イメージ", font=font(GOTHIC, 24), fill=PALE)
    f = font(MINCHO, 58)
    text_lines(d, [[("入らなかった家具は、", BG)], [("窓からクレーンで。", BG)]],
               1390, 200, f, DARK, appear(t, 0.3), dy=20)
    text_lines(d, [[("2万〜5万円", GOLD)]], 1390, 410, font(MINCHO, 150), DARK, appear(t, 1.6), dy=30)
    text_lines(d, [[("しかも分かるのは、", PALE)], [("いつも", PALE), ("買った後", BG), ("。", PALE)]],
               1390, 680, font(GOTHIC_B, 48), DARK, appear(t, 3.6), dy=20)


def _box_poly(cx, cy, L, Wd, ang):
    c, s = math.cos(ang), math.sin(ang)
    pts = [(-L / 2, -Wd / 2), (L / 2, -Wd / 2), (L / 2, Wd / 2), (-L / 2, Wd / 2)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def _dashed_poly(d, pts, col, width=4, dash=18):
    for a, b in zip(pts, pts[1:] + pts[:1]):
        n = max(1, int(math.dist(a, b) / dash))
        for i in range(0, n, 2):
            p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
            q = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
            d.line([p, q], fill=col, width=width)


def s_float(img, d, t):
    text_lines(d, [[("いまの計算は、家具が", INK), ("宙に浮いて", TERRA), ("動く前提。", INK)]],
               W / 2, 70, font(MINCHO, 62), BG, appear(t, 0.2), dy=20)
    text_lines(d, [[("でも実際は、人が持って運ぶ。人の体も通り道をふさぐ。", SUB)]],
               W / 2, 170, font(GOTHIC_B, 40), BG, appear(t, 1.6), dy=12)
    a = appear(t, 0.8)
    # 左: ふわふわ浮いて、ひとりでに回る家具(点線)
    cx, cy = 520, 560 + 18 * math.sin(t * 2.2)
    ang = t * 0.9
    d.ellipse([400, 800, 640, 830], fill=mix((214, 205, 184), BG, a))
    _dashed_poly(d, _box_poly(cx, cy, 300, 110, ang), mix(SUB, BG, a))
    text_lines(d, [[("いまの計算", SUB)]], 520, 300, font(GOTHIC_B, 40), BG, a)
    text_lines(d, [[("宙に浮いて、ひとりでに回る", SUB)]], 520, 900, font(GOTHIC, 36), BG, a)
    # 右: 2人が両端を持って廊下を進む
    a = appear(t, 2.4)
    if a > 0:
        col_wall = mix(INK, BG, a)
        d.line([(1240, 330), (1240, 840)], fill=col_wall, width=6)
        d.line([(1560, 330), (1560, 840)], fill=col_wall, width=6)
        y = 760 - clamp((t - 2.4) / 4.0) * 260
        d.rectangle([1360, y - 150, 1440, y + 150], fill=mix(FURN, BG, a), outline=col_wall, width=3)
        for yy in (y - 195, y + 195):
            d.ellipse([1370, yy - 30, 1430, yy + 30], fill=mix(CARRIER, BG, a), outline=col_wall, width=3)
        text_lines(d, [[("Odoriba", OK)]], 1400, 260, font(MINCHO, 46), BG, a)
        text_lines(d, [[("運ぶ人の体まで、計算に入れる", OK)]], 1400, 900,
                   font(GOTHIC, 36), BG, a)


def s_compare(img, d, t):
    data = DATA
    text_lines(d, [[("動きで見ると", INK), (f"（例：幅{data['stair_w']}cmの階段に、タンス {data['furn']}cm）", SUB)]],
               W / 2, 50, font(MINCHO, 48), BG, appear(t, 0.0, 0.4))
    speed = 1.35            # 計算のコマ1つを何フレームで見せるか
    i = max(0, (t - 0.8) * FPS)
    nA, nB = len(data["alone"]), len(data["carried"])
    iA, iB = min(nA - 1, int(i / speed)), min(nB - 1, int(i / speed))
    stuck_t = 0.8 + (nB - 1) * speed / FPS
    done_t = 0.8 + (nA - 1) * speed / FPS
    stuck = t >= stuck_t
    pw, ph = 860, 760
    for k, (title, col_t, x0) in enumerate([("家具だけ", OK, 70), ("2人で持って運ぶ", NG, 990)]):
        d.rectangle([x0, 150, x0 + pw, 150 + ph + 110], fill=(251, 250, 246), outline=(217, 210, 195), width=2)
        d.text((x0 + 30, 170), title, font=font(MINCHO, 46), fill=col_t)
    left = stair_panel(data, pw - 40, ph - 80, [(data["alone"][iA], FURN, None)])
    img.paste(left, (90, 240))
    blink_red = stuck and int((t - stuck_t) * 3) % 2 == 0 or (stuck and t - stuck_t > 2.0)
    fr = data["carried"][iB]
    right = stair_panel(
        data, pw - 40, ph - 80,
        [(fr, NG if blink_red else FURN, CARRIER_NG if stuck else CARRIER)],
        label="← ここで詰まる" if stuck else None,
        label_pos=(data["stair"]["w"] + 30, data["stair"]["landy1"] + 32))
    img.paste(right, (1010, 240))
    a = appear(t, done_t, 0.4)
    if a > 0:
        text_lines(d, [[("○ 通る（余裕 ", OK), (f"{data['margin_alone']}cm", OK), ("）", OK)]],
                   70 + pw / 2, 905, font(MINCHO, 60), (251, 250, 246), a)
    a = appear(t, stuck_t, 0.4)
    if a > 0:
        text_lines(d, [[(f"× 踊り場で あと {data['short_carried']}cm", NG)]],
                   990 + pw / 2, 905, font(MINCHO, 60), (251, 250, 246), a)


@lru_cache(maxsize=None)
def _photo(name, h):
    im = Image.open(PHOTOS / name).convert("RGB")
    return im.resize((int(im.width * h / im.height), h), Image.LANCZOS)


def s_box(img, d, t):
    for k, (name, x) in enumerate([("danbo-ru_nagasa.jpg", 110), ("tumari1.jpg", 470)]):
        a = appear(t, 0.2 + 0.5 * k, 0.7)
        if a <= 0:
            continue
        ph = _photo(name, 760)
        if name == "tumari1.jpg":
            ph = ph.crop((0, 0, min(ph.width, 520), 760))
        y = int(160 + 60 * (1 - a))
        tile = Image.blend(Image.new("RGB", ph.size, BG), ph, a)
        img.paste(tile, (x, y))
    f = font(MINCHO, 64)
    text_lines(d, [[("まず、", INK), ("本物の箱", TERRA), ("で", INK)], [("確かめた。", INK)]],
               1080, 200, f, BG, appear(t, 1.2), anchor="left", dy=16)
    f = font(GOTHIC_B, 46)
    text_lines(d, [[("箱だけなら通るのに、", INK)], [("人が持つと", INK), ("通らない場所", NG), ("が", INK)],
                   [("本当にあった。", INK)]], 1080, 470, f, BG, appear(t, 2.4), anchor="left", dy=16)
    text_lines(d, [[("段ボール箱 60×42cm ・ 2026年8月", SUB)]], 1080, 800, font(GOTHIC, 32), BG,
               appear(t, 3.0), anchor="left")


def s_cta(img, d, t):
    text_lines(d, [[("このブースで、", BG)], [("試せます。", (251, 236, 214))]], W / 2, 120,
               font(MINCHO, 100), TERRA, appear(t, 0.2), dy=24)
    steps = ["階段をえらぶ", "家具をえらぶ", "すぐ結果が出る"]
    for k, s in enumerate(steps):
        a = appear(t, 1.2 + 0.4 * k, 0.5)
        if a <= 0:
            continue
        x = 330 + k * 450
        d.rounded_rectangle([x, 470, x + 400, 680], radius=16, fill=mix((160, 64, 52), TERRA, a))
        d.text((x + 200, 535), "①②③"[k], font=font(MINCHO, 64), fill=mix(BG, TERRA, a), anchor="mm")
        d.text((x + 200, 625), s, font=font(GOTHIC_B, 42), fill=mix(BG, TERRA, a), anchor="mm")
    text_lines(d, [[("あなたの家の階段の寸法でも、どうぞ。", BG)]], W / 2, 750, font(GOTHIC_B, 46),
               TERRA, appear(t, 2.6))
    a = appear(t, 3.0)
    if a > 0:
        y = 880 + 14 * math.sin(t * 5)
        d.polygon([(W / 2 - 40, y), (W / 2 + 40, y), (W / 2, y + 50)], fill=mix(BG, TERRA, a))
    text_lines(d, [[("試作品の計算による目安です", (240, 205, 190))]], W / 2, 980, font(GOTHIC, 28),
               TERRA, appear(t, 3.0))


def s_overview(img, d, t):
    text_lines(d, [[("Odoriba", DARK), ("（おどりば）は、", INK)]], W / 2, 170, font(MINCHO, 60), BG,
               appear(t, 0.1), dy=16)
    text_lines(d, [[("スマホで測って、", INK)], [("運ぶ人ごと", TERRA), ("計算する。", INK)]], W / 2, 290,
               font(MINCHO, 88), BG, appear(t, 0.5), dy=20)
    for k, s in enumerate(["スキャン", "寸法を読み取る", "運ぶ人ごと計算"]):
        a = appear(t, 1.3 + 0.35 * k, 0.4)
        if a <= 0:
            continue
        x = 300 + k * 460
        d.rounded_rectangle([x, 700, x + 400, 830], radius=14, fill=mix(DARK, BG, a))
        d.text((x + 200, 765), f"{'①②③'[k]} {s}", font=font(GOTHIC_B, 40), fill=mix(BG, DARK, a),
               anchor="mm")
        if k < 2:
            d.text((x + 430, 765), "→", font=font(GOTHIC_B, 44), fill=mix(BRASS, BG, a), anchor="mm")


# ---- スキャンした実際の階段(点群)を描く ----
SCAN = None


def load_scan(path, per_face=4, seed=0):
    """スキャンOBJを、cm・Z-up・中心原点の色つき点群として読む。
    軸と単位の合わせ方は scan_demo.load_scan_zup と同じ。寸法は measure_stairs で測る。
    頂点だけだとまばらで階段に見えないので、各三角形の中にも点を撒き、
    色はテクスチャから取り、面の向きで陰影をつけて立体感を出す。"""
    import trimesh
    import measure_stairs as ms
    m = trimesh.load(path, force="mesh")
    tex = np.asarray(m.visual.material.image.convert("RGB"), np.float32)
    uv = np.asarray(m.visual.uv, np.float32)
    m.apply_scale(100.0)
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces)
    rng = np.random.default_rng(seed)
    # 各三角形に重心座標でランダムに点を撒く(頂点そのものも含める)
    b = rng.random((len(f), per_face, 2)).astype(np.float32)
    flip = b.sum(-1) > 1
    b[flip] = 1 - b[flip]
    w = np.concatenate([1 - b.sum(-1, keepdims=True), b], -1)          # (F, k, 3)
    pts = np.einsum("fkj,fjc->fkc", w, v[f]).reshape(-1, 3)
    puv = np.einsum("fkj,fjc->fkc", w, uv[f]).reshape(-1, 2)
    fn = np.repeat(np.asarray(m.face_normals, np.float32), per_face, axis=0)
    pts = np.concatenate([pts, v])
    puv = np.concatenate([puv, uv])
    nrm = np.concatenate([fn, np.asarray(m.vertex_normals, np.float32)])
    th, tw = tex.shape[:2]
    tx = np.clip((puv[:, 0] % 1.0) * (tw - 1), 0, tw - 1).astype(int)
    ty = np.clip((1 - puv[:, 1] % 1.0) * (th - 1), 0, th - 1).astype(int)
    cols = tex[ty, tx]
    # 斜め上からの光で陰影(平らな面は明るく、壁は少し暗く)
    light = np.array([0.35, -0.45, 0.82], np.float32)
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.6 * np.abs(nrm @ light)
    cols = np.clip(cols * shade[:, None] * 1.15, 0, 255)
    pts -= (pts.min(0) + pts.max(0)) / 2
    horiz = np.abs(nrm[:, 2]) > 0.9
    r = ms.measure(path)
    band = np.floor((pts[:, 2] - pts[:, 2].min()) / r["rise"]).astype(int) % 2
    hl = np.where(band[:, None] == 0, np.array(GOLD, np.float32), np.array(TERRA, np.float32))
    return {"v": pts, "cols": cols, "horiz": horiz, "hl": hl, "dims": r}


def cloud_image(w, h, yaw_deg, elev_deg, highlight, bg):
    """点群を斜め上から見た画像。highlight(0..1)で平らな面(段・床)を色づけする。"""
    v, cols = SCAN["v"], SCAN["cols"]
    if highlight > 0:
        cols = cols.copy()
        hz = SCAN["horiz"]
        cols[hz] = cols[hz] * (1 - highlight) + SCAN["hl"][hz] * highlight
    yaw, el = np.radians(yaw_deg), np.radians(elev_deg)
    x = v[:, 0] * np.cos(yaw) - v[:, 1] * np.sin(yaw)
    y = v[:, 0] * np.sin(yaw) + v[:, 1] * np.cos(yaw)
    z = v[:, 2]
    up = y * np.sin(el) + z * np.cos(el)
    depth = y * np.cos(el) - z * np.sin(el)
    k = min(w, h) / 680.0
    sx = (w / 2 + x * k).astype(int)
    sy = (h / 2 - up * k).astype(int)
    order = np.argsort(-depth)            # 奥から手前へ塗る
    sx, sy, c = sx[order], sy[order], cols[order].astype(np.uint8)
    arr = np.empty((h, w, 3), np.uint8)
    arr[:] = bg
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ok = (sx + dx >= 0) & (sx + dx < w) & (sy + dy >= 0) & (sy + dy < h)
        arr[sy[ok] + dy, sx[ok] + dx] = c[ok]
    return Image.fromarray(arr)


SCAN_YAW0, SCAN_SPIN = -35.0, 7.0      # 回転の初期角(度)と速さ(度/秒)


def _step_chip(d, x, y, label, bg_col, fg_col):
    f = font(GOTHIC_B, 40)
    w = f.getlength(label) + 60
    d.rounded_rectangle([x, y, x + w, y + 70], radius=35, fill=bg_col)
    d.text((x + w / 2, y + 35), label, font=f, fill=fg_col, anchor="mm")


def s_scan(img, d, t):
    img.paste(cloud_image(1100, 1000, SCAN_YAW0 + SCAN_SPIN * t, 24, 0.0, DARK), (30, 30))
    a = appear(t, 0.3)
    _step_chip(d, 1180, 170, "① スマホでスキャン", mix(GOLD, DARK, a), DARK)
    text_lines(d, [[("iPhoneの", BG), ("LiDAR", GOLD), ("で、", BG)], [("階段を撮るだけ。", BG)]],
               1180, 300, font(MINCHO, 62), DARK, appear(t, 0.9), anchor="left", dy=16)
    text_lines(d, [[("LiDAR ＝ 光で距離を測るセンサー。", PALE)], [("撮った階段が、立体のデータになる。", PALE)]],
               1180, 520, font(GOTHIC_B, 36), DARK, appear(t, 2.2), anchor="left", dy=12)
    text_lines(d, [[("← 実際にスキャンした家の階段（Scaniverseで撮影）", PALE)]], 1180, 820,
               font(GOTHIC, 28), DARK, appear(t, 3.4), anchor="left")


def s_dims(img, d, t):
    t0 = SCENES_BY_FN[s_scan]          # 前の場面から回転を続ける
    hl = appear(t, 0.3, 1.2)
    img.paste(cloud_image(1100, 1000, SCAN_YAW0 + SCAN_SPIN * (t0 + t), 24, hl, DARK), (30, 30))
    a = appear(t, 0.2)
    _step_chip(d, 1180, 110, "② 寸法を読み取る", mix(GOLD, DARK, a), DARK)
    text_lines(d, [[("平らな面（段・床）を拾って、", BG)], [("自動で計算する。", BG)]], 1180, 220,
               font(MINCHO, 50), DARK, appear(t, 0.6), anchor="left", dy=12)
    r = SCAN["dims"]
    n = int(r["n_steps_total"])
    rows = [("階段の幅", f"{r['width']:.0f}cm"), ("1段の高さ", f"{r['rise']:.0f}cm"),
            ("1段の奥行き", f"{r['tread']:.0f}cm"), ("段数", f"{n // 2}段＋{n - n // 2}段"),
            ("天井の高さ", f"{r['headroom']:.0f}cm")]
    for k, (name, val) in enumerate(rows):
        a = appear(t, 1.4 + 0.5 * k, 0.4)
        if a <= 0:
            continue
        y = 420 + k * 92
        d.text((1180, y), name, font=font(GOTHIC_B, 40), fill=mix(PALE, DARK, a))
        d.text((1780, y), val, font=font(MINCHO, 52), fill=mix(GOLD, DARK, a), anchor="ra")
        d.line([(1180, y + 70), (1780, y + 70)], fill=mix((60, 90, 75), DARK, a), width=2)
    text_lines(d, [[("見た目ではなく、寸法を測る道具として使う", PALE)]], 1180, 900, font(GOTHIC, 30),
               DARK, appear(t, 4.2), anchor="left")


def s_result(img, d, t):
    a = appear(t, 0.1)
    _step_chip(d, 120, 70, "③ 運ぶ人ごと計算する", mix(TERRA, BG, a), BG)
    r = SCAN["dims"]
    text_lines(d, [[(f"この階段（幅{r['width']:.0f}cm）に、ソファ（200×90×85cm）を運び込むと…", INK)]],
               120, 190, font(MINCHO, 50), BG, appear(t, 0.5), anchor="left", dy=10)
    cards = [("家具だけ", "○ 通る", "踊り場で 余裕 8cm", OK, 0.9, 140),
             ("2人で持って運ぶ", "× 通らない", "踊り場で あと 29cm", NG, 2.0, 1000)]
    for title, big, small, col, st, x in cards:
        a = appear(t, st, 0.5)
        if a <= 0:
            continue
        d.rectangle([x, 300, x + 780, 760], fill=mix((251, 250, 246), BG, a), outline=mix(col, BG, a), width=3)
        d.rectangle([x, 300, x + 780, 314], fill=mix(col, BG, a))
        d.text((x + 390, 390), title, font=font(MINCHO, 50), fill=mix(SUB, BG, a), anchor="mm")
        d.text((x + 390, 540), big, font=font(MINCHO, 120), fill=mix(col, BG, a), anchor="mm")
        d.text((x + 390, 680), small, font=font(GOTHIC_B, 48), fill=mix(col, BG, a), anchor="mm")
    text_lines(d, [[("同じソファ・同じ階段でも、", INK), ("人が運ぶと考えるだけで答えが逆になる。", TERRA)]],
               W / 2, 840, font(MINCHO, 52), BG, appear(t, 3.4), dy=12)


SCENES = [
    (4.0, DARK, s_hook),
    (6.0, DARK, s_crane),
    (7.0, BG, s_float),
    (4.0, BG, s_overview),
    (8.0, DARK, s_scan),
    (7.0, DARK, s_dims),
    (7.0, BG, s_result),
    (12.0, BG, s_compare),
    (5.0, BG, s_box),
    (5.0, TERRA, s_cta),
]
SCENES_BY_FN = {fn: dur for dur, _, fn in SCENES}
TOTAL = sum(s[0] for s in SCENES)
FADE = 0.4
DATA = None


def render_scene(i, t):
    dur, bg, fn = SCENES[i]
    img = Image.new("RGB", (W, H), bg)
    fn(img, ImageDraw.Draw(img), t)
    return img


def frame_at(T):
    """全体の時刻T(秒)のコマ。場面の終わりFADE秒は次の場面の頭とクロスフェードする。"""
    T %= TOTAL
    start = 0.0
    for i, (dur, bg, _) in enumerate(SCENES):
        if T < start + dur:
            break
        start += dur
    t = T - start
    img = render_scene(i, t)
    if t > dur - FADE:
        nxt = render_scene((i + 1) % len(SCENES), 0.0)
        img = Image.blend(img, nxt, ease((t - (dur - FADE)) / FADE))
        bg = SCENES[(i + 1) % len(SCENES)][1] if t > dur - FADE / 2 else bg
    # 右下の名前と、下端の進行バー
    d = ImageDraw.Draw(img)
    on_dark = sum(bg) < 400
    d.text((W - 40, H - 34), "Odoriba（おどりば）", font=font(MINCHO, 30),
           fill=PALE if on_dark else BRASS, anchor="rm")
    d.rectangle([0, H - 6, int(W * T / TOTAL), H], fill=GOLD if on_dark else BRASS)
    return img


def main():
    global DATA
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--still", type=float, help="この秒のコマだけPNGに書き出す")
    ap.add_argument("--scan", default=str(SCAN_OBJ), help="階段スキャンのOBJ(Scaniverseの書き出し)")
    args = ap.parse_args()
    DATA = A.build_data()
    global SCAN
    SCAN = load_scan(args.scan)

    if args.still is not None:
        p = Path(args.out).with_name(f"odoriba_still_{args.still:g}.png")
        frame_at(args.still).save(p)
        print(f"書き出し: {p}")
        return

    import imageio_ffmpeg
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = int(TOTAL * FPS)
    w = imageio_ffmpeg.write_frames(str(out), (W, H), fps=FPS, codec="libx264",
                                    quality=8, pix_fmt_out="yuv420p", macro_block_size=8)
    w.send(None)
    for f in range(n):
        w.send(np.asarray(frame_at(f / FPS)))
        if f % (FPS * 5) == 0:
            print(f"  {f / FPS:4.0f} / {TOTAL:.0f} 秒", flush=True)
    w.close()
    print(f"書き出し: {out}  ({TOTAL:.0f}秒・{n}コマ)")


if __name__ == "__main__":
    main()
