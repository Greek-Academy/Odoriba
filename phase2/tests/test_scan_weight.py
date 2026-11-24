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


@pytest.mark.parametrize("space_cls", [_OpenSpace, _LowDoorway])
def test_tilt_detail_gives_the_same_bottleneck_as_the_plain_sweep(box, space_cls):
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


def test_tilt_detail_tries_the_same_poses_whatever_the_weight(box):
    # 60kgなら約29度まで傾けられる。試す傾きは格子のまま(0度と15度)なので、
    # 結果は「上限15度・重さなし」と同じになる。重さごとに試す傾きを足すと、
    # 重い方が重さ不明より良く出ることがある
    box.setattr(L, "FURN_WEIGHT_KG", 60.0)
    cap = L.max_tilt_deg(2)
    assert 15.0 < cap < 30.0
    result = _detail(_LowDoorway())
    assert result["bottleneck"]["max_tilt_deg"] == cap
    by_tilt = {e["max_tilt_deg"]: e for e in result["bottleneck_by_tilt"]}
    assert list(by_tilt) == list(sp.SWEEP_PITCHES)
    assert result["bottleneck"]["profile"] == by_tilt[15.0]["profile"]


def test_heavier_never_has_more_room(box, space_cls=_LowDoorway):
    worst = []
    for weight in (None, 30.0, 60.0, 79.0):
        box.setattr(L, "FURN_WEIGHT_KG", weight)
        worst.append(_detail(space_cls())["bottleneck"]["capacity_cm"])
    # 重さなし(上限55度)より、30kg(垂直まで傾けられる)の方が余裕があってよい
    assert worst[1:] == sorted(worst[1:], reverse=True)


def test_tilt_detail_restores_the_weight_and_tilt_limit(box):
    box.setattr(L, "FURN_WEIGHT_KG", 70.0)
    before = (L.FURN_WEIGHT_KG, L.MAX_TILT_DEG)
    _detail(_LowDoorway())
    assert (L.FURN_WEIGHT_KG, L.MAX_TILT_DEG) == before
