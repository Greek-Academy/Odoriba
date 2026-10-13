"""ボトルネック掃引の基準値の回帰テスト。

オラクルの変更で「あと何cm」が黙って変わるのを防ぐ。値は docs/decisions.md
(2026-10-10, 体と家具の干渉を足した後)の表と同じ。家具200x50x65cm、
運搬者2人、刻み15cmの掃引。意図して値を変えたときは、decisions.md に理由を
書いてからここを更新する。

実行(リポジトリの phase2/ で): python -m pytest tests  (1分ほどかかる)
"""

import numpy as np
import pytest

import phase2_lstair as L
import phase2_ustair as U
import phase2_winder as W

# (名前, 環境を作る関数, 掃引関数, 家具単体の最小余裕, 運搬者2人の最小余裕)
CASES = [
    ("lstair", L.build_lstairs, L.sweep_capacity, 8.0, -32.9),
    ("ustair", U.build_ustairs, U.sweep_capacity, 8.8, -32.9),
    ("winder", W.build_winder, W.sweep_capacity, 5.3, -28.9),
]


def _min_capacity(sweep):
    # 運搬者の立つ床がない位置(-inf)は除く(bottleneck_entry・sensitivity と同じ扱い)
    return min(c for _, c, _ in sweep if np.isfinite(c))


@pytest.mark.parametrize("name,build,sweep,solo,carried", CASES,
                         ids=[c[0] for c in CASES])
def test_sweep_baseline(name, build, sweep, solo, carried):
    outer, obstacles = build()
    assert _min_capacity(sweep(outer, obstacles, with_human=False)) == \
        pytest.approx(solo, abs=0.05)
    assert _min_capacity(sweep(outer, obstacles, num_carriers=2, with_human=True)) == \
        pytest.approx(carried, abs=0.05)
