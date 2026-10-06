"""
可搬性オラクルの仮定ごとの感度分析: 「どの仮定を外すと不足量が何cm変わるか」。

L字・折り返し・回り階段の3つで、運搬者2人のボトルネック掃引を
条件を変えて回し、最も狭い位置の余裕(cm、負なら不足)を一覧にする。

きっかけ: 3つの階段で不足量がそろって29.0cmになり、内訳を見ると決め手は
運搬者の体ではなく「傾き上限55度のせいで家具自身が壁に当たる」ことだった。
pitchの主張(人体の占有が失敗の主因)がどの条件で成り立つかを数字で確かめる。

条件:
  家具単体     運搬者なし・傾き自由(浮遊する剛体。既存プランナーの見方)
  既定         傾き上限55度・体・手の届く範囲すべて有効
  傾き70度     傾き上限だけ70度に緩める
  傾き上限なし 傾き上限を90度にする(垂直に立ててよい)
  体なし       運搬者の体の占有を無視(手の届く範囲・傾き上限は有効)
  届く範囲なし 持つ高さの制約を無視
  体・届く範囲なし 傾き上限55度だけ残す
  傾き自由・届く範囲なし 運搬者の体だけ残す(体の占有が単独で効くかを見る)
  (円柱)が付く列 体の判定モデルを人型から従来の円柱に戻した同条件
              (体のモデルを変えた影響を同じ表で比べる)

実行(リポジトリの phase2/ で、数分かかる):
    python sensitivity.py                    # results/sensitivity.json に保存
"""

import json
import os
import time

import numpy as np

import phase2_lstair as L
import phase2_ustair as U
import phase2_winder as W

# (列名, 運搬者あり, 傾き上限, 有効にするオラクルの項, 体の判定モデル)
CONDITIONS = [
    ("家具単体", False, 90.0, {"reach": True, "body": True}, "humanoid"),
    ("既定", True, 55.0, {"reach": True, "body": True}, "humanoid"),
    ("傾き70度", True, 70.0, {"reach": True, "body": True}, "humanoid"),
    ("傾き上限なし", True, 90.0, {"reach": True, "body": True}, "humanoid"),
    ("体なし", True, 55.0, {"reach": True, "body": False}, "humanoid"),
    ("届く範囲なし", True, 55.0, {"reach": False, "body": True}, "humanoid"),
    ("体・届く範囲なし", True, 55.0, {"reach": False, "body": False}, "humanoid"),
    ("傾き自由・届く範囲なし", True, 90.0, {"reach": False, "body": True}, "humanoid"),
    ("既定(円柱)", True, 55.0, {"reach": True, "body": True}, "cylinder"),
    ("傾き上限なし(円柱)", True, 90.0, {"reach": True, "body": True}, "cylinder"),
    ("傾き自由・届く範囲なし(円柱)", True, 90.0, {"reach": False, "body": True}, "cylinder"),
]

# (名前, 環境を作る関数, 掃引関数, 歩行面関数)
STAIRS = [
    ("L字", L.build_lstairs,
     lambda o, ob, wh: L.sweep_capacity(o, ob, num_carriers=2, with_human=wh), None),
    ("折り返し", U.build_ustairs,
     lambda o, ob, wh: U.sweep_capacity(o, ob, num_carriers=2, with_human=wh), U.floor_z),
    ("回り", W.build_winder,
     lambda o, ob, wh: W.sweep_capacity(o, ob, num_carriers=2, with_human=wh), W.floor_z),
]


def run_condition(build, sweep, floor_fn, with_human, max_tilt, terms, body_model):
    """1条件ぶんの掃引を回し、最も狭い位置の余裕と内訳を返す。
    L のモジュール変数を一時的に書き換え、終わったら必ず戻す。"""
    orig_tilt, orig_terms = L.MAX_TILT_DEG, dict(L.ORACLE_TERMS)
    orig_body = L.BODY_MODEL
    try:
        L.MAX_TILT_DEG = max_tilt
        L.BODY_MODEL = body_model
        L.ORACLE_TERMS.update(terms)
        outer, obstacles = build()
        res = [r for r in sweep(outer, obstacles, with_human) if np.isfinite(r[1])]
        s, c, pose = min(res, key=lambda r: r[1])
        bd = None
        if with_human:
            bd = L.clearance_breakdown(np.array(pose[0]), np.array(pose[1]),
                                       outer, obstacles, 2, floor_fn=floor_fn)
        return {"s": s, "capacity_cm": c, "breakdown": bd}
    finally:
        L.MAX_TILT_DEG = orig_tilt
        L.BODY_MODEL = orig_body
        L.ORACLE_TERMS.clear()
        L.ORACLE_TERMS.update(orig_terms)


def main():
    print(f"家具 {L.FURN_L:.0f}x{L.FURN_W:.0f}x{L.FURN_H:.0f}cm / 運搬者2人 / "
          "値 = 最も狭い位置での最良姿勢の余裕(cm、負なら不足)\n")
    table = {}
    for name, build, sweep, floor_fn in STAIRS:
        table[name] = {}
        for col, with_human, max_tilt, terms, body_model in CONDITIONS:
            t0 = time.time()
            r = run_condition(build, sweep, floor_fn, with_human, max_tilt, terms,
                              body_model)
            table[name][col] = r
            limiting = (L.BREAKDOWN_LABELS[r["breakdown"]["limiting"]]
                        if r["breakdown"] else "-")
            print(f"  {name:6s} {col:10s} {r['capacity_cm']:+7.1f}cm  決め手: {limiting}"
                  f"  [{time.time() - t0:.0f}s]")
        print()

    # Markdownの表(docsに貼れる形)
    cols = [c[0] for c in CONDITIONS]
    print("| 階段 | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name in table:
        print(f"| {name} | " + " | ".join(f"{table[name][c]['capacity_cm']:+.1f}"
                                          for c in cols) + " |")

    out = os.path.join("results", "sensitivity.json")
    os.makedirs("results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"furniture": [L.FURN_L, L.FURN_W, L.FURN_H], "num_carriers": 2,
                   "table": table}, f, indent=1, ensure_ascii=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
