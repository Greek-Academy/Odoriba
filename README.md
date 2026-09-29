# Odoriba

引越し搬入支援アプリ。家具が玄関から目的の部屋まで**運び込めるか**を判定する。
プロジェクトの全体像は `CLAUDE.md`、詳細は `docs/` 以下を参照。

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
python scan_demo.py <scan.obj>              # 実スキャン階段に家具を置いてクリアランスを可視化
python scan_furniture.py --selftest        # スキャン家具の通過判定の自己テスト
python scan_furniture.py <furniture.obj>    # スキャンした家具がL字階段を通れるか判定
python scan_furniture.py <furniture.obj> --gif out.gif  # 通り抜けるアニメGIFを書き出す
python scan_furniture.py <furniture.obj> --html out.html # 回せる3Dで通り抜けを書き出す
python scan_combined.py <stair.obj> <box.obj>   # 実スキャン家具×実スキャン階段の組み合わせ3D
python scan_babylon.py <stair.obj> <box.obj>    # Babylon.js版(テクスチャ半透明・見た目重視)
python virtual_demo.py --width 90 --rise 18 --tread 25 --steps 8 --furniture 150 45 40 --carriers 2 --html out.html  # 数値から仮想階段を組んで家具を通す
python measure_stairs.py stair.obj              # スキャンから階段寸法を計測
python stair_report.py --measure-stair stair.obj          # あなたの階段への答え(ボトルネック/上限サイズ)
python stair_report.py --measure-stair stair.obj --quick    # 掃引のみ(速い)
python virtual_demo.py --measure-stair stair.obj --furniture-scan box.obj --html out.html  # 計測値で仮想階段を組んで通す
python catalog.py                           # 定番家具7種の一斉判定表(15〜30分。--table-onlyで保存済み結果の表示のみ)
```

保存済みJSONからの**図の再生成(数秒、探索は走らない)**:

```bash
cd phase2
python phase2_lstair_viz.py    # 比較PNG (phase2_lstair.png)
python phase2_lstair_gif.py    # アニメーションGIF (phase2_lstair.gif)
python phase2_lstair_3d.py     # 3Dビューア (phase2_lstair_3d.html; ブラウザで開いて回す)
```

画像・GIFはgit管理しない(`.gitignore`)。JSONがコミットされているので、
クローン直後でも上の2コマンドだけで全図を再生成できる。

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
python phase2_ustair_viz.py                              # 説明用の1枚図(日本語、L字との比較つき) phase2_ustair.png
```

`phase2_ustair_viz.py` の図は、初めて見る人向けに「①家具だけ(上から見た図) /
②2人で運ぶ(詰まる姿勢を赤で) / ③入口からの道のりと余裕のグラフ」を1枚に
まとめたもの。日本語フォント(Meiryo / BIZ UDPGothic 等)が必要。

既定寸法(幅90・6+6段・家具200x50x65)の結果: 家具単体はPASS(最も狭い踊り場で
余裕8.8cm)、運搬者2人はBLOCKED。踊り場の入口から出口まで(弧長490〜685cm)
一度も余裕がプラスに戻らず、最大29cm不足。L字(同じ家具で29cm不足)は踊り場の
出口で余裕が戻るのに対し、折り返しは踊り場全体が詰まり所になる。
