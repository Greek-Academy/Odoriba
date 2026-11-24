"""スキャン空間の掃引(scan_plan)に重さを渡したときのテスト。

重さは「どこまで傾けてよいか」にだけ効く(decisions.md 2026-10-13)。
ここでは壁が遠い平らな空間を使い、傾きと重さの扱いだけを確かめる。
"""

import numpy as np
import pytest

import phase2_lstair as L
import scan_plan as sp


class _Centerline:
    """+x 方向へまっすぐ 320cm 進む、平らな中心線(掃引は7か所で済む)。"""

    s_total = 320.0
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


def test_too_heavy_to_hold_flat_is_reported_as_weight(box):
    # 2人で水平に持つと1人100kg。業者の上限(40kg)を超えるので、どの姿勢でも持てない。
    # 以前は「運搬者の立つ床が見つからない」という違う理由のエラーになっていた
    box.setattr(L, "FURN_WEIGHT_KG", 200.0)
    with pytest.raises(sp.TooHeavyError) as e:
        _judge()
    assert "1人100kg" in str(e.value) and "上限40kg" in str(e.value)
    assert "床" not in str(e.value)


def test_weight_that_can_be_held_flat_is_not_an_error(box):
    box.setattr(L, "FURN_WEIGHT_KG", 60.0)   # 水平なら1人30kg
    assert _judge()["bottleneck"]["capacity_cm"] > 0


# ---- 傾きごとの掃引結果(tilt_detail) ----

class _LowDoorway(_OpenSpace):
    """s=160cm 付近に、高さ40cmまでしか通れない低い所がある空間。

    高さ60cmの箱はどう傾けても当たる。余裕が傾きごとに変わる例として使う。
    """

    def clearance_points(self, pts):
        pts = np.asarray(pts)
        near = np.abs(pts[:, 0] - 160.0) < 20.0
        return np.where(near, 40.0 - pts[:, 2], 100.0)


def _detail(space):
    return sp.judge(space, num_carriers=2, rrt=False, tilt_detail=True,
                    log=lambda *_: None)


@pytest.mark.parametrize("weight", [None, 70.0])
@pytest.mark.parametrize("space_cls", [_OpenSpace, _LowDoorway])
def test_tilt_detail_gives_the_same_bottleneck_as_the_plain_sweep(box, weight, space_cls):
    box.setattr(L, "FURN_WEIGHT_KG", weight)
    plain = sp.judge(space_cls(), num_carriers=2, rrt=False, log=lambda *_: None)
    detail = _detail(space_cls())
    for key in ("s", "capacity_cm", "pose", "profile", "no_pose_s", "limit", "contact"):
        assert detail["bottleneck"][key] == plain["bottleneck"][key], key
    assert detail["bottleneck_furniture_only"] == plain["bottleneck_furniture_only"]


def test_tilt_detail_lists_each_tilt_limit_in_order(box):
    by_tilt = _detail(_OpenSpace())["bottleneck_by_tilt"]
    assert [e["max_tilt_deg"] for e in by_tilt] == list(sp.SWEEP_PITCHES)
    # 傾きの上限を広げると試せる姿勢が増えるので、余裕は減らない
    caps = [e["capacity_cm"] for e in by_tilt]
    assert caps == sorted(caps)


def test_tilt_detail_adds_the_weight_limited_tilt_to_the_list(box):
    box.setattr(L, "FURN_WEIGHT_KG", 70.0)
    cap = L.max_tilt_deg(2)
    assert 0.0 < cap < 90.0 and cap not in sp.SWEEP_PITCHES
    result = _detail(_OpenSpace())
    assert result["bottleneck"]["max_tilt_deg"] == cap
    assert cap in [e["max_tilt_deg"] for e in result["bottleneck_by_tilt"]]


def test_tilt_detail_restores_the_weight_and_tilt_limit(box):
    box.setattr(L, "FURN_WEIGHT_KG", 70.0)
    before = (L.FURN_WEIGHT_KG, L.MAX_TILT_DEG)
    _detail(_LowDoorway())
    assert (L.FURN_WEIGHT_KG, L.MAX_TILT_DEG) == before
