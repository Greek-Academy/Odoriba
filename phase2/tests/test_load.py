"""重さ(下側の人の負担)のモデルの単体テスト。

支える点を端面の下側の辺に置くと、ロールなしで長軸を θ 傾けたときの
下側の人の割合は 1/2 + (H / 2L)·tanθ になる(H=家具の高さ、L=長さ)。
"""

import contextlib

import numpy as np
import pytest

import phase2_lstair as L


@contextlib.contextmanager
def weight(kg, preset="mover", com=(0.0, 0.0, 0.0)):
    """重さの設定を一時的に変え、終わったら必ず戻す。"""
    orig = (L.FURN_WEIGHT_KG, L.CARRIER_PRESET, L.COM_OFFSET)
    L.FURN_WEIGHT_KG, L.CARRIER_PRESET, L.COM_OFFSET = kg, preset, com
    try:
        yield
    finally:
        L.FURN_WEIGHT_KG, L.CARRIER_PRESET, L.COM_OFFSET = orig


def _quat(pitch):
    return L._pose(0.0, pitch).as_quat()


def test_no_weight_keeps_fixed_tilt_limit():
    assert L.max_tilt_deg() == L.MAX_TILT_DEG
    assert not L.with_tilt_violation(_quat(50.0))
    assert L.with_tilt_violation(_quat(60.0))


def test_level_furniture_splits_evenly():
    with weight(60.0):
        assert L.carrier_loads_kg(_quat(0.0), 2) == pytest.approx([30.0, 30.0])


@pytest.mark.parametrize("pitch", [15.0, 30.0, 45.0])
def test_lower_carrier_share_matches_formula(pitch):
    with weight(60.0):
        loads = L.carrier_loads_kg(_quat(pitch), 2)
    frac = 0.5 + (L.FURN_H / (2 * L.FURN_L)) * np.tan(np.radians(pitch))
    assert max(loads) == pytest.approx(60.0 * frac)
    assert sum(loads) == pytest.approx(60.0)


def test_low_center_of_mass_eases_lower_carrier():
    with weight(60.0):
        centered = max(L.carrier_loads_kg(_quat(40.0), 2))
    with weight(60.0, com=(0.0, 0.0, -L.FURN_H / 2)):
        low = max(L.carrier_loads_kg(_quat(40.0), 2))
    assert low == pytest.approx(30.0)
    assert low < centered


def test_upright_furniture_rests_on_lower_carrier():
    with weight(60.0):
        assert max(L.carrier_loads_kg(_quat(90.0), 2)) == pytest.approx(60.0)


def test_single_carrier_holds_everything():
    # 1人なら傾きに関係なく全部を持つ。上限以下なら垂直まで、超えれば運べない
    with weight(30.0):
        assert L.carrier_loads_kg(_quat(30.0), 1) == [30.0]
        assert L.max_tilt_deg(1) == 90.0
    with weight(50.0):
        assert L.max_tilt_deg(1) is None


def test_max_tilt_from_weight():
    # 60kgを業者2人(1人40kg): 割合2/3まで -> tanθ = (2/3 - 1/2)·2L/H
    with weight(60.0):
        cap = L.max_tilt_deg(2)
        assert cap == pytest.approx(np.degrees(np.arctan((1 / 6) * 2 * L.FURN_L / L.FURN_H)),
                                    abs=1e-6)
        assert not L.with_tilt_violation(_quat(cap - 0.5), 2)
        assert L.with_tilt_violation(_quat(cap + 0.5), 2)
    # 同じ60kgでも自分たちで運ぶ(1人25kg)なら水平でも持てない
    with weight(60.0, preset="diy"):
        assert L.max_tilt_deg(2) is None
        assert L.tilt_pitches((0.0, 15.0), 2) == []
    # 軽ければ垂直まで立てられる
    with weight(30.0):
        assert L.max_tilt_deg(2) == 90.0


def test_tilt_pitches_adds_the_cap():
    assert L.tilt_pitches((0.0, 15.0, 30.0, 45.0, 55.0, 70.0)) == [0.0, 15.0, 30.0, 45.0, 55.0]
    with weight(60.0):
        cap = L.max_tilt_deg()
        assert L.tilt_pitches((0.0, 15.0, 30.0, 45.0, 55.0, 70.0)) == \
            [0.0, 15.0, 30.0, 45.0, cap]


def test_weight_limit_is_inverse_of_tilt_cap():
    assert L.weight_limit_kg(0.0, 2) == pytest.approx(80.0)
    assert L.weight_limit_kg(0.0, 2, preset="diy") == pytest.approx(50.0)
    assert L.weight_limit_kg(60.0, 1) == pytest.approx(40.0)
    w = L.weight_limit_kg(40.0, 2)
    with weight(w):
        assert L.max_tilt_deg(2) == pytest.approx(40.0, abs=1e-6)


def test_find_needed_tilt_restores_settings():
    # 37度以上傾けられれば通る、という仮の判定
    needed = L.find_needed_tilt(lambda: L.MAX_TILT_DEG >= 37.0, tol=0.5)
    assert 37.0 <= needed <= 37.5
    assert L.find_needed_tilt(lambda: False) is None
    assert L.find_needed_tilt(lambda: True) == 0.0
    assert L.MAX_TILT_DEG == 55.0 and L.FURN_WEIGHT_KG is None
