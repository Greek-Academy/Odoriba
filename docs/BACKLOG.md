# BACKLOG — 自律実行の作業キュー

スケジュール実行されるクラウドエージェント（および人間）が上から順に消化する作業キュー。

## 運用ルール（エージェントはここに従うこと）

- **1回の実行で消化するのは、未完了(`[ ]`)の最上位タスク1つだけ**
- 作業は CLAUDE.md の規約に従う: develop から `feature/<説明>` を切る／Conventional Commits／日本語コメント／細かいコミット粒度／`merge --no-ff` で develop へ統合して push
- 完了したら該当行を `[x]` にし、末尾に `(YYYY-MM-DD, ブランチ名)` を追記して、その変更もコミットに含める
- タスクが技術的に不可能・前提が壊れていると判明した場合は `[!]` にして理由を1行書き、次のタスクには進まない（人間の判断待ち）
- 探索（RRT）を回す場合は `--max-iter` を絞った動作確認をしてから本番実行する

## キュー

- [ ] **実スキャンの取り込み仕上げ(継続)**: 一次デモ(配置・可視化)と水密化ロジック(to_free_space_voxel)まで実装済み。残り — **より完全なスキャンの取得**(現状のstair1は壁が閉じておらず囲まれた空間を作れない=水密化不可と判明。壁・床・天井を囲うように撮り直すか手動で蓋付け)、進行方向の軸合わせ自動化、START/GOAL定義、RRTでの経路探索、**メジャー実測との精度突き合わせ**(1フライト分の実測が必要)
  - 分かったこと: stair1スキャンは高さ約5.1m=2フライト分を捉えている。仮実測(235cm=1フライト)とは範囲が違うため、精度検証はスキャンと同じ範囲の実測が要る
- [ ] **螺旋階段・S字階段の環境**: 回転した直方体で環境を組めるようにし、S字（L字の逆向き接続）とコーナー踏み面（回り階段）のどちらか一方をデモ化する。`horizontal_clearance_points` のAABB前提もこのとき拡張する
- [x] **メッシュ→SDF受け口**: `build_lstairs` の環境をメッシュ（OBJ/PLY）として書き出し、trimesh で読み込んで点→メッシュ距離で衝突判定する `validator` を作り、既存の直方体方式と判定結果が一致することを確認する（iPhone LiDAR スキャンの受け入れ準備。open3d が入らない環境では trimesh のみで可） (2026-09-08, feature/mesh-sdf-input)
- [x] **GIF・3Dビューアへの数字の反映**: 比較PNGにある「2人なら長さ162cmまで」「踊り場で余裕8cm」の注記を、GIF と 3Dビューア（HTML）にも入れて3つの成果物の情報を揃える (2026-09-08, feature/viz-sync-numbers)
- [x] **`_nearest` のKD-tree化**: 探索の反復回数を増やしたときのO(n^2)を解消する（scipy.spatial.cKDTree。SE(3)距離の回転項は近似でよい。導入前後で同一seedの結果が変わらないことを確認） (2026-09-08, feature/nearest-vectorize: KD-treeではなくnumpyベクトル化で対応。同一距離式のまま8.8s->3.7s。KD-treeは反復数万の規模で再検討)
- [x] **`p.W_ROT` 上書きの引数化**: `phase2_lstair.py` が import 時に `phase2_demo.W_ROT` を書き換えている副作用を、`rrt_connect` の引数（または設定オブジェクト）に置き換える (2026-09-08, feature/wrot-param)
- [x] **ヒアリング質問リストのドキュメント化**: 引っ越し業者への質問を `docs/interview-questions.md` に整理する。傾きの限界（仮定55度）・持てる高さ（仮定10〜190cm）・踊り場での回し方・「長さ何cmまでならいける」の現場判断（シミュレータの162cmの答え合わせ）を必ず含める

## 完了済み

- [x] 定番家具カタログの一斉判定(catalog.py、7種の一覧表) (2026-09-08, feature/furniture-catalog)
- [x] スキャン取り込み前処理(scan_ingest.py: 単位推定・穴埋め・ボクセル水密化、MeshEnvの非水密フォールバック) (2026-09-19, feature/scan-ingest)
- [x] 実スキャン階段の一次デモ(scan_demo.py: cm正規化・Z-up軸合わせ・配置と運搬シーケンス可視化) (2026-09-19, feature/scan-demo)
- [x] スキャンの水密化ロジック(to_free_space_voxel: 囲まれた空気の抽出、閉じ判定。閉じた箱で検証済み。実スキャン階段は壁が閉じず適用不可と判明) (2026-09-19, feature/scan-watertight)
- [x] スキャン家具を運ぶ剛体として判定(scan_furniture.py: 床除去で対象物切り出し・OBB正規化・点群でL字階段の通過判定。実スキャンの段ボールでPASS確認) (2026-09-19, feature/scan-furniture)

- [x] L字階段（踊り場）の環境・保持拘束付きオラクル・可視化 (2026-09-01, feature/l-shaped-staircase)
- [x] 運搬者ありの上限長さ二分探索 --find-max-length（2人なら162cmまで） (2026-09-08, feature/max-length-search)
- [x] 3Dビューア plotly→HTML (2026-09-08, feature/lstair-3d-viewer)
- [x] 通る場合の「余裕◯cm」出力（家具単体の掃引、踊り場で8.0cm） (2026-09-08, feature/lateral-clearance)
