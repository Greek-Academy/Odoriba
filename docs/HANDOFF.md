# 引き継ぎメモ（別セッション用） — Odoriba

最終更新: 2026-10-06（スキャン空間での判定を追加）

このファイルは「別のClaude/開発セッションに作業を引き継ぐ」ための現状まとめ。
コードの詳細は各ファイルのdocstringとCLAUDE.md・docs/を見ること。

## プロジェクト一言
家具が玄関→部屋まで「運び込めるか」を判定するアプリ（名前=Odoriba/踊り場）。
核心の主張: 既存は「浮いた剛体」しか見ないが、実際は人が運ぶ＝人体も通路を占有する。
「家具だけなら通るが、人が運ぶと踊り場で回れない」を、通る/通らない＋あと何cm＋上限サイズで出す。

## 現在地（フェーズ）
- Phase 0 実測ゲート: 完了
- Phase 1 2D・L字廊下(グリッドBFS): 完了 (`phase1/`)
- Phase 2 3D・階段＋可搬性オラクル＋RRT-Connect: **進行中（ここが今の主戦場）**
- Phase 3 Goバックエンド(API/DB/ジョブキュー・非同期): 未着手
- Phase 4 Swift自前スキャン(ARKit): 未着手

全部Pythonの研究プロトタイプ段階。バックエンド/フロントのアプリはまだ無い。

## 何が動くか（Phase 2 の到達点）
1. 仮想L字階段の判定: `phase2_lstair.py`（環境=SDF直方体の合併、可搬性オラクル、RRT-Connect）
   - 出力: 通る/通らない、ボトルネック掃引で「あと何cm」、上限サイズ二分探索（幅/長さ）
   - 可視化: `phase2_lstair_viz.py`(PNG)、`_gif.py`(GIF)、`_3d.py`(plotly HTML)
2. 数値から仮想階段を組んで家具を通す: `virtual_demo.py`
   - 例: `python virtual_demo.py --width 90 --rise 18 --tread 25 --steps 8 --furniture 150 45 40 --carriers 2 --html out.html`
3. スキャン取り込み系（iPhone LiDAR / Scaniverse で撮ったOBJ）:
   - `scan_ingest.py`: 単位推定・穴埋め・水密化ロジック（`to_free_space_voxel`）
   - `scan_furniture.py`: 家具スキャンを切り出して（床除去）通過判定＋3D/GIF可視化
   - `measure_stairs.py`: 階段スキャンから寸法計測（蹴上げ/踏み面/幅/段数/天井）
   - `scan_demo.py` / `scan_combined.py` / `scan_babylon.py`: 実スキャンの可視化（点群/面/Babylon）
4. あなたの階段への答えレポート: `stair_report.py`
   - `python stair_report.py --measure-stair stair.obj --quick`（掃引のみ・速い）
5. 定番家具カタログ一斉判定: `catalog.py`（重い。普段は1家具ずつでよい）
6. 運搬者の体は人型（`human_figure.py`）。描画も判定も人型。円柱は `phase2_lstair.BODY_MODEL="cylinder"` で比較用に残してある
7. **スキャンした空間をそのまま判定**（型の階段なしで、玄関→部屋の経路全体）:
   - `scan_space.py`: メッシュ→距離場（2cm格子）・床・中心線。水密化不要。撮れた床の範囲で区切る
   - `scan_plan.py`: 中心線に沿ったRRTとボトルネック掃引。一番厳しい点（何に当たったか）も返す
   - `route_env.py`: 検証用の合成経路（玄関・廊下・曲がり角・ドア）。`python route_env.py` で正解と突き合わせ
   - 使い方: `sp = ScanSpace(mesh, 自由空間の1点); sp.build_centerline(出発点, 目的地); scan_plan.sweep_capacity(sp, num_carriers=2)`
     （ScanSpaceはオラクルの `outer` にそのまま渡せる。obstacles=None、floor_fn=sp.floor_fn）

## 重要な設計判断（触る前に知っておくこと）
- **状態空間に人の姿勢を足さない**。6DOF(位置3+クォータニオン4)のまま、衝突判定器を
  可搬性オラクルに差し替える（`feasible=collisionFree ∧ ∃h carriable`）。ここが新規性。
- 環境はSDF（「あと何cm」が副産物で出る）。家具は表面サンプル点で近似。
- RRTのSE(3)距離の回転重みは `w_rot`。**L字では60**（0.4だと90°旋回のエッジ検証が
  粗くて壁を貫通する誤判定。過去のバグ）。`rrt_connect(..., w_rot=...)` で渡す。
- スキャンは**撮ったまま環境として使う**（`scan_space`）。手すり・扉などの意味は認識しない。
  外すものは外して撮る前提で、出力は「どこで何cm足りないか」と当たった点の座標。
  水密化はせず、出発点からの塗りつぶし＋撮れた床の範囲で内外を決める。
  寸法から型の階段を組む方式（`measure_stairs`）も残っている。
- 「あと何cm」は測る経路と掃引の刻みで変わる（L字で手引きの中心線 −29cm / 自動の中心線 約−13cm、
  刻みによる揺れ約2cm）。発表の数字の扱いは未決定（`docs/decisions.md`）。

## 実データの現状（stair1スキャン）
- 計測結果: 幅107cm・蹴上げ20cm・踏み面26cm・天井241cm・13+13段（全高520cm）
- 蹴上げ/段数/全高は安定。踏み面/幅は水平面ベースで改善したが数cm誤差は残る。
- 精度の実測（メジャー）突き合わせは**未実施**（ユーザーが実測できておらず、以前の値は仮）。
- 例の判定: ソファ200×90×85 → 家具だけ踊り場で余裕8cm / 2人で29cm不足。
- `scan_space` で直接判定するのは**撮り直しが必要**: 階段の踏み面がほとんど写っていない
  （1階から踊り場まで歩いてつながらない）。撮り直しは階段を上りながら踏み面にカメラを向ける。
- ファイルの場所（リポジトリ外）: `C:\dev\alpha-project\Scaniverse 2026-09-19 113017 stair1\`

## 既知の未解決・次の候補（docs/BACKLOG.md も参照）
- 螺旋・S字階段（回転OBBで環境を組む。BACKLOGの残り）
- スキャン精度の実測突き合わせ（5cm超なら手入力に切替、というゲート）
- 引っ越し業者ヒアリング（`docs/interview-questions.md` に質問リスト準備済み）
- スキャン空間の判定の続き: 詰まった所を赤く表示する3Dビューア、出発点・目的地の自動提案
  （床をつないだ図で歩いて一番遠い2点）、掃引の刻みを細かくする、stair1 の撮り直し

## 運用ルール（CLAUDE.md準拠）
- Git Flow: developから `feature/<kebab>` を切り、`merge --no-ff` でdevelopへ、push。
  マージ後もブランチは消さない。コミットはConventional Commits・日本語コメント・細かい粒度。
- 生成物(png/gif/html/obj/glb)と `demo/` はgit管理外（`.gitignore`）。結果はJSONで保存し図は再生成。
- クラウド運用: Greek-Academy組織へのGitHub App承認待ち（未承認だと `claude --cloud` の
  push不可・ルーチン作成不可）。承認後にBACKLOG消化ルーチンを作る計画（メモリ参照）。

## 環境
- Windows。`python`(3.13) に numpy/scipy/matplotlib/pillow/plotly/trimesh 導入済み。
- 実行は `cd phase2` してから各スクリプト。RRTを回すときは `--max-iter` を絞って動作確認→本番。

## デモ成果物の置き場
`C:\dev\Odoriba\demo\`（比較PNG・GIF・3D HTML・Babylon・スライド素材 `SLIDES_BRIEF.md` 等、23ファイル）。
git管理外なのでローカルのみ。図はコード＋results JSONから再生成可能。

## いま進行中の派生作業
- 発表スライド: `demo/SLIDES_BRIEF.md`（デスクトップ版Claudeに渡して.pptx化する想定）
- Qiita記事: 技術厚めの下書きをチャットで作成済み（未保存。必要なら `demo/qiita_draft.md` に保存）
