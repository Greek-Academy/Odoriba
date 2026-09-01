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
python phase2_lstair.py --find-max-width   # 家具単体で通る最大幅を二分探索してJSONに追記
```

保存済みJSONからの**図の再生成(数秒、探索は走らない)**:

```bash
cd phase2
python phase2_lstair_viz.py    # 比較PNG (phase2_lstair.png)
python phase2_lstair_gif.py    # アニメーションGIF (phase2_lstair.gif)
```

画像・GIFはgit管理しない(`.gitignore`)。JSONがコミットされているので、
クローン直後でも上の2コマンドだけで全図を再生成できる。
