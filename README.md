# Odoriba

引越し搬入支援アプリ。家具が玄関から目的の部屋まで**運び込めるか**を判定する。
プロジェクトの全体像は `CLAUDE.md`、詳細は `docs/` 以下を参照。

## PRD
プロダクトのターゲット・ユースケース・ゴール/ノンゴールなどの情報はdocs/PRD.mdを参照。

## セットアップ

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pillow
```

(trimesh / open3d は Phase 2 後半のスキャンメッシュ処理用。現時点のデモには不要)

## Phase 1 — 2D・L字廊下

```bash
cd phase1
python phase1_demo.py    # 比較PNG (phase1_demo.png) + 臨界幅の探索
python phase1_gif.py     # アニメーションGIF (phase1_demo.gif)
```

## Phase 2 — 3D・階段

```bash
cd phase2
python phase2_demo.py    # 直線階段。数値結果のみ(RRT-Connectの動作確認)
```

### L字階段(踊り場つき)

探索の実行(RRTで数分かかる)。結果は `results/lstair_result.json` に保存され、
このJSONはコミットする:

```bash
cd phase2
python phase2_lstair.py                    # RRT 2ケース + ボトルネック掃引
python phase2_lstair.py --find-max-width    # 家具単体で通る最大幅を二分探索してJSONに追記
python phase2_lstair.py --find-max-length   # 運搬者ありで通る最大長さを二分探索してJSONに追記
python mesh_env.py                          # メッシュ受け口と直方体方式の判定一致チェック(LiDAR受け入れ準備)
python scan_ingest.py --selftest            # スキャン取り込み前処理の自己テスト(実スキャン不要)
python scan_ingest.py <scan.obj>            # 実スキャンOBJ/PLYを水密メッシュに整えて書き出す
python measure_stairs.py stair.obj              # スキャンから階段寸法を計測
python stair_report.py --measure-stair stair.obj          # あなたの階段への答え(ボトルネック/上限サイズ)
python stair_report.py --measure-stair stair.obj --quick    # 掃引のみ(速い)
python stair_report.py --width 90 --rise 18 --tread 25 --steps 8 --quick  # 寸法を数値で直接
python catalog.py                           # 定番家具7種の一斉判定表(15〜30分。--table-onlyで保存済み結果の表示のみ)
```

保存済みJSONからの**3Dビューアの再生成(数秒、探索は走らない)**:

```bash
cd phase2
python phase2_lstair_3d.py     # 3Dビューア (phase2_lstair_3d.html; ブラウザで開いて回す)
```

HTMLはgit管理しない(`.gitignore`)。JSONがコミットされているので、
クローン直後でもこのコマンドだけで再生成できる。

### 折り返し(U字)階段

日本の戸建てで最も多い、踊り場で180度折り返す階段。家具形状・可搬性オラクル・
RRTはL字のものをそのまま使い、幾何(環境・歩行面・中心線・サンプラー)だけを
`phase2_ustair.py` が持つ。結果は `results/ustair_result.json` に保存する:

```bash
cd phase2
python phase2_ustair.py --max-iter 800 --skip-sweep     # 動作確認(十数秒)
python phase2_ustair.py --html ustair_3d.html            # RRT 2ケース + ボトルネック掃引 + 3D
python phase2_ustair.py --json results/ustair_result.json --html ustair_3d.html  # 保存済みJSONから3Dだけ再生成
python phase2_ustair.py --furniture 180 60 40 --width 80  # 家具・階段幅を変えて判定
```

既定寸法(幅90・6+6段・家具200x50x65)の結果: 家具単体はPASS(最も狭い踊り場で
余裕8.8cm)、運搬者2人はBLOCKED。踊り場の入口から出口まで(弧長490〜685cm)
一度も余裕がプラスに戻らず、最大29cm不足。L字(同じ家具で29cm不足)は踊り場の
出口で余裕が戻るのに対し、折り返しは踊り場全体が詰まり所になる。
