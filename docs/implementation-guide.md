# Odoriba 実装ガイド（Week 1-4）

各タスクの「どうやるか」と、**時間を食う落とし穴**。

```bash
pip install numpy scipy shapely matplotlib trimesh
```

**2Dでは shapely を使う。** 自分で幾何演算を書くと、それだけで数日消える。

---

# Week 1 — 2Dで動くものを作る

## 1-1. 環境をポリゴンで定義（4h）

### 発想の転換：障害物ではなく「自由空間」を持つ

障害物を並べて「どれとも当たらないか」を判定するより、**通れる領域そのものを1つのポリゴンとして持つ**ほうが圧倒的に楽。

```python
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# L字廊下 = 2つの矩形の和
W = 1.2  # 廊下幅
free_space = unary_union([
    box(0, 0, W, 4.0),      # 縦の廊下
    box(0, 4.0 - W, 4.0, 4.0)  # 横の廊下
])
```

衝突判定がこの1行になる。

```python
def is_free(shape):
    return free_space.contains(shape)
```

**さらに、余裕（クリアランス）も一発で出る。**

```python
clearance = free_space.exterior.distance(shape)  # 壁までの最短距離
```

「あと4cm」がここから直接得られる。この設計にしておくと、後で SDF に移行する必要すら薄い。

### 描画

```python
import matplotlib.pyplot as plt

def draw(ax, free_space, shapes):
    x, y = free_space.exterior.xy
    ax.fill(x, y, color='#e8e8e8')
    for s in shapes:
        sx, sy = s.exterior.xy
        ax.fill(sx, sy, alpha=0.6)
    ax.set_aspect('equal')
```

**この描画関数を最初に作る。** 以降のデバッグが全部これに乗る。

---

## 1-2. 長方形の衝突判定（4h）

```python
import numpy as np
from shapely.affinity import rotate, translate

def make_rect(x, y, theta, length, width):
    r = box(-length/2, -width/2, length/2, width/2)
    r = rotate(r, theta, origin=(0,0), use_radians=True)
    return translate(r, x, y)
```

判定そのものは10分で終わる。**残りの時間は動作確認に使う。**

```python
# 手で数パターン置いて、目で見て正しいか確かめる
for (x, y, th) in [(0.6, 1.0, 0), (0.6, 1.0, np.pi/2), (0.6, 3.5, np.pi/4)]:
    print(x, y, th, is_free(make_rect(x, y, th, 1.8, 0.6)))
```

**ここで手を抜くと、RRTが解を返さないときに「探索が悪いのか判定が悪いのか」が分からなくなる。** 必ず目視で確認しておく。

---

## 1-3. RRT-Connect 実装（8h）

### 構造

```python
class Node:
    def __init__(self, q, parent=None):
        self.q = q          # (x, y, theta)
        self.parent = parent

def rrt_connect(start, goal, is_valid, max_iter=20000):
    ta, tb = [Node(start)], [Node(goal)]
    for i in range(max_iter):
        q_rand = sample()
        new = extend(ta, q_rand, is_valid)
        if new is not None:
            if connect(tb, new.q, is_valid):
                return build_path(ta, tb, new)
        ta, tb = tb, ta        # 交互に伸ばす
    return None
```

### 落とし穴①：距離関数 ← ここで8時間の半分が溶ける

**位置と角度は単位が違う。** そのまま足すと壊れる。

```python
def angle_diff(a, b):
    d = (a - b + np.pi) % (2*np.pi) - np.pi   # -π〜π に正規化
    return d

def dist(q1, q2, w=0.4):
    dx, dy = q1[0]-q2[0], q1[1]-q2[1]
    dth = angle_diff(q1[2], q2[2])
    return np.hypot(dx, dy) + w * abs(dth)
```

**`w` の決め方**: 家具の外接円半径（対角線の半分）にする。1ラジアン回すと角がおよそ `r` だけ動くので、それが自然な換算になる。

1.8×0.6 のソファなら `r ≈ 0.95`。ただし小さめ（0.3〜0.5）から始めて、解が出るか見ながら調整するほうが実用的。

**解が返らないときは、まずここを疑う。**

### 落とし穴②：エッジの検証

端点だけ見ると、**壁を突き抜ける経路が「有効」になる。**

```python
def edge_valid(q1, q2, is_valid, res=0.02):
    n = max(2, int(dist(q1, q2) / res))
    for i in range(n+1):
        if not is_valid(interpolate(q1, q2, i/n)):
            return False
    return True
```

`res` は 2cm 程度。細かすぎると遅く、粗すぎると壁抜けする。

### 落とし穴③：角度の補間

```python
def interpolate(q1, q2, t):
    x = q1[0] + t*(q2[0]-q1[0])
    y = q1[1] + t*(q2[1]-q1[1])
    th = q1[2] + t*angle_diff(q2[2], q1[2])   # 短いほうに回る
    return (x, y, th)
```

`q1[2] + t*(q2[2]-q1[2])` と書くと、**359度から1度へ行くのに358度ぶん回る。**

### 落とし穴④：ゴール判定

完全一致は起きない。許容範囲で判定する。

```python
def reached(q, goal, tol_pos=0.05, tol_ang=0.1):
    return (np.hypot(q[0]-goal[0], q[1]-goal[1]) < tol_pos
            and abs(angle_diff(q[2], goal[2])) < tol_ang)
```

### 解が返らないときの診断順

1. **木を描画する** ← 最優先。どこで止まっているかが一目で分かる
2. 距離関数の `w` を変える（0.1 〜 1.0 で振る）
3. `steer` の刻み幅を小さくする
4. ゴールバイアス（5〜10%の確率でゴールをサンプルする）を入れる
5. 廊下を広げてみる（環境が本当に解けないだけかもしれない）

**「解が無い」のか「見つけられない」のかを、必ず切り分ける。**

---

## 1-4. アニメーション（5h）

```python
from matplotlib.animation import FuncAnimation

def animate(path, free_space, L, Wd, out='out.gif'):
    fig, ax = plt.subplots(figsize=(6,6))
    def frame(i):
        ax.clear()
        draw(ax, free_space, [make_rect(*path[i], L, Wd)])
        ax.set_title(f'step {i}/{len(path)}')
    anim = FuncAnimation(fig, frame, frames=len(path), interval=50)
    anim.save(out, writer='pillow')
```

**経路は補間して滑らかにしてから描く。** RRTの生の出力はガタガタで、見栄えが悪い。

---

# Week 2 — 人を入れる ← 本命

## 2-1. 人カプセルの配置（6h）

2Dなので、人は**円**でいい。

```python
from shapely.geometry import Point

R_HUMAN = 0.25   # 人の半径
L_ARM   = 0.45   # 腕の長さ

def place_humans(x, y, theta, L, Wd, side_offset=0.0):
    """家具の両端に把持点を置き、その外側に人を立たせる"""
    c = np.array([x, y])
    d = np.array([np.cos(theta), np.sin(theta)])   # 家具の長軸方向
    n = np.array([-d[1], d[0]])                    # 法線方向

    grip1 = c + d * (L/2)
    grip2 = c - d * (L/2)

    h1 = grip1 + d * L_ARM + n * side_offset
    h2 = grip2 - d * L_ARM + n * side_offset
    return Point(*h1).buffer(R_HUMAN), Point(*h2).buffer(R_HUMAN)
```

`side_offset` は「横にずれて立つ」余地。オラクルで複数試す。

---

## 2-2. 可搬性判定（6h）

```python
SIDE_OFFSETS = [-0.3, -0.15, 0.0, 0.15, 0.3]

def carriable(x, y, theta, L, Wd):
    rect = make_rect(x, y, theta, L, Wd)
    if not is_free(rect):
        return False, 0.0

    best = -1.0
    for off in SIDE_OFFSETS:
        h1, h2 = place_humans(x, y, theta, L, Wd, off)
        if is_free(h1) and is_free(h2):
            c = min(free_space.exterior.distance(h1),
                    free_space.exterior.distance(h2))
            best = max(best, c)
    return best >= 0, best
```

**プランナー本体は一切変更しない。** `is_valid` に渡す関数を差し替えるだけ。

```python
path_free  = rrt_connect(start, goal, lambda q: is_free(make_rect(*q, L, Wd)))
path_human = rrt_connect(start, goal, lambda q: carriable(*q, L, Wd)[0])
```

**この2行が、この企画の全てです。**

---

## 2-3. 差が出るシナリオを作り込む（8h）

探索的な作業だが、**当てずっぽうにやらない。廊下幅で二分探索する。**

```python
def find_critical_width(L, Wd, lo=0.7, hi=2.5, iters=12):
    """自由剛体は通るが人ありでは通らない、廊下幅の範囲を探す"""
    results = []
    for _ in range(iters):
        w = (lo + hi) / 2
        env = build_L_corridor(w)
        ok_free  = rrt_connect(..., is_free_only)  is not None
        ok_human = rrt_connect(..., carriable_only) is not None
        results.append((w, ok_free, ok_human))
        if ok_free and not ok_human:
            return w, results          # 見つけた
        elif not ok_free:
            lo = w                     # 狭すぎ
        else:
            hi = w                     # 広すぎ（両方通る）
    return None, results
```

### 出発点の数値

| | 値 |
|---|---|
| 家具（ソファ想定） | 1.8m × 0.6m |
| 人の半径 | 0.25m |
| 腕の長さ | 0.45m |
| 廊下幅の探索範囲 | 0.9m 〜 2.0m |

**うまくいかないときに動かす順番**

1. **家具を長くする**（1.8 → 2.1m）— 差が出やすくなる
2. **曲がり角を直角以外にする** — 斜めの角は人が回り込みにくい
3. **人の半径を上げる**（0.25 → 0.3m）— 冬服・靴を想定すれば妥当

### 見つかったら必ず記録する

```
廊下幅 1.15m / ソファ 1.8×0.6 / 人 r=0.25
  自由剛体  : 通る（324ステップ）
  人あり    : 通らない（20000回で解なし）
  詰まる箇所: 曲がり角の内側、θ≈50° 付近で後方の人が壁に接触
```

**これがピッチの中身そのもの。** スクリーンショットも撮っておく。

---

## 2-4. 左右分割デモ（5h）

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
# 左: 自由剛体の経路（通る）
# 右: 人ありの探索（詰まる位置で停止）
```

右側は**詰まった瞬間で止めて、接触箇所を赤く塗る。** そこに箱の実験の動画を並べれば、デモが完成する。

---

# Week 3 — 3Dへ拡張

## 3-1. SE(3)の表現（6h）

**オイラー角を使わない。** ジンバルロックと補間の問題で必ず詰まる。

```python
from scipy.spatial.transform import Rotation, Slerp

# 状態 = (位置3, クォータニオン4)
def interpolate_se3(q1, q2, t):
    pos = q1[:3] + t*(q2[:3] - q1[:3])
    key = Rotation.from_quat([q1[3:], q2[3:]])
    rot = Slerp([0,1], key)(t).as_quat()
    return np.concatenate([pos, rot])

def dist_se3(q1, q2, w=0.4):
    dp = np.linalg.norm(q1[:3] - q2[:3])
    r1, r2 = Rotation.from_quat(q1[3:]), Rotation.from_quat(q2[3:])
    dth = (r1.inv() * r2).magnitude()      # 回転角（ラジアン）
    return dp + w * dth
```

### 一様なランダム回転

素朴にオイラー角を一様サンプルすると**偏る**。

```python
rot = Rotation.random()   # scipyが正しくやってくれる
```

---

## 3-2. 3D衝突判定（8h）

**全部を直方体（OBB）で表す。** 階段も家具も。そうすれば分離軸定理（SAT）で済む。

```python
def obb_intersect(c1, R1, e1, c2, R2, e2):
    """中心・回転行列・半径ベクトルの2つのOBBが交差するか
       15本の軸で分離を調べる（3+3+9）"""
    axes = list(R1.T) + list(R2.T)
    for a in R1.T:
        for b in R2.T:
            c = np.cross(a, b)
            if np.linalg.norm(c) > 1e-8:
                axes.append(c / np.linalg.norm(c))
    for ax in axes:
        r1 = sum(abs(np.dot(ax, R1.T[i])) * e1[i] for i in range(3))
        r2 = sum(abs(np.dot(ax, R2.T[i])) * e2[i] for i in range(3))
        if abs(np.dot(ax, c2 - c1)) > r1 + r2:
            return False      # 分離軸が見つかった
    return True
```

100行未満、依存ライブラリなし、十分速い。**FCLやOMPLを入れる必要はまだない。**

人カプセルは**円柱**として、カプセル対OBBの距離で判定する。

---

## 3-3. 階段環境の構築（6h）

**実測した数字から組み立てる。**

```python
def build_stairs(width, rise, tread, n_steps, landing_w, landing_d):
    boxes = []
    for i in range(n_steps):
        boxes.append(make_box(
            center=(width/2, tread*(i+0.5), rise*i - 0.1),
            size=(width, tread, 0.2)))
    boxes.append(make_box(...))   # 踊り場
    boxes += build_walls(...)      # 壁・天井・手すり
    return boxes
```

**手すりを忘れない。** 壁から張り出している分が、実際に一番効く。

---

## 3-4. 解が返らないときの対処（5h）

2Dで動いても、3Dの階段では**まず解が出ません。** 想定内。

対処の順番:

1. **木を3Dで描画する** — どこで止まっているか
2. 距離関数の `w` を振る
3. ゴールバイアスを上げる（10〜20%）
4. **骨格誘導を入れる** — 階段の中心線に沿って位置をサンプルし、姿勢だけランダムにする

骨格誘導の最小版:

```python
def sample_guided(centerline, p=0.7):
    if np.random.rand() < p:
        t = np.random.rand()
        pos = interpolate_centerline(centerline, t)
        pos += np.random.normal(0, 0.15, 3)   # 少し散らす
    else:
        pos = sample_uniform_position()
    return np.concatenate([pos, Rotation.random().as_quat()])
```

**これが効くかどうかが Week 3 の山場。** 効かなければ Week 4 を使う。

---

# Week 4 — 実測と突き合わせる

## 4-1. 比較表を作る（6h）

箱の実験の各サイズについて、3手法の予測を並べる。

| 箱サイズ | 外接直方体 | 自由剛体 | 人あり | **実際** |
|---|---|---|---|---|
| 160×60×70 | ○ | ○ | ○ | ○ |
| 180×60×70 | ○ | ○ | ○ | ○ |
| 190×60×70 | ○ | ○ | **✕** | **✕** |
| 200×60×70 | ○ | **✕** | ✕ | ✕ |

**3行目が全て**です。自由剛体は「通る」と言い、人ありは「通らない」と言い、現実は後者だった——という行が1つでもあれば、主張は実証されています。

## 4-2. ずれの分析（8h）

一致しなかった場合、原因を切り分ける。

| 症状 | 疑うところ |
|---|---|
| 予測は通る、現実は通らない | 手すり・幅木の測り忘れ／人の半径が小さすぎ |
| 予測は通らない、現実は通る | 人の立ち位置の自由度が足りない（`SIDE_OFFSETS` を増やす） |
| 全体的にずれる | 環境の寸法入力ミス。まず測り直し |

**「合わなかった」も結果です。** なぜ合わなかったかを説明できれば、それも成果になる。

## 4-3. 記録（7h）

- 各手法の予測と実測の対応表
- 通り抜けのアニメーション（2D / 3D）
- 左右分割デモ
- 箱の実験の動画
- **うまくいかなかった条件の記録** ← 後で必ず必要になる

---

# 全体の注意

- **各週の終わりに、必ず何か動いている状態にする。** 飛び飛びで作業するので、再開コストを最小にする
- **可視化を先に作る。** これを後回しにすると、数字だけ眺めて何時間も溶ける
- **Week 2 で止まっても目的は達成。** 3Dはスケールさせる作業で、主張自体はそこで証明済み
- **OMPL / FCL / Go / Swift は、この4週間では触らない**
