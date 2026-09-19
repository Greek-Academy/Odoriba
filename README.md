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
