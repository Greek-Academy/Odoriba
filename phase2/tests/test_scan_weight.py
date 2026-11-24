"""スキャン空間の掃引(scan_plan)に重さを渡したときのテスト。

重さは「どこまで傾けてよいか」にだけ効く(decisions.md 2026-10-13)。
ここでは壁が遠い平らな空間を使い、傾きと重さの扱いだけを確かめる。
"""

import numpy as np
import pytest

import phase2_lstair as L
import scan_plan as sp


class _Centerline:
    """+x 方向へまっすぐ 600cm 進む、平らな中心線。"""

    s_total = 600.0
    corners = []

    def xy_at(self, s):
        return np.array([float(s), 0.0])

    def heading_at(self, s):
        return 0.0

    def slope_at(self, s):
        return 0.0

    def z_at(self, s):
        return 0.0

    def corner_at(self, s):
        return None


class _OpenSpace:
    """壁が遠くにしか無い、平らな床だけの空間(床の高さは0)。"""

    centerline = _Centerline()

    def clearance_points(self, pts):
        return np.full(len(pts), 100.0)

    def floor_fn(self, x, y, z_hint=None):
        return 0.0


@pytest.fixture
def box(monkeypatch):
    """100×45×60cm の箱を2人で運ぶ設定にする。"""
    monkeypatch.setattr(L, "FURN_L", 100.0)
    monkeypatch.setattr(L, "FURN_W", 45.0)
    monkeypatch.setattr(L, "FURN_H", 60.0)
    monkeypatch.setattr(L, "COM_OFFSET", (0.0, 0.0, 0.0))
    monkeypatch.setattr(L, "CARRIER_PRESET", "mover")
    return monkeypatch


def _judge():
    return sp.judge(_OpenSpace(), num_carriers=2, rrt=False, log=lambda *_: None)


def test_open_space_passes_without_weight(box):
    assert _judge()["bottleneck"]["capacity_cm"] > 0


@pytest.mark.xfail(strict=True, reason="重さが理由なのに、立つ床が無いという説明になる")
def test_too_heavy_to_hold_flat_is_reported_as_weight(box):
    # 2人で水平に持つと1人100kg。業者の上限(40kg)を超えるので、どの姿勢でも持てない
    box.setattr(L, "FURN_WEIGHT_KG", 200.0)
    with pytest.raises(ValueError) as e:
        _judge()
    assert "重すぎ" in str(e.value)
    assert "床" not in str(e.value)
