"""階ごとの間取り図(scan_floorplan)のテスト。

実物のスキャンは使わず、「この高さにこの広さの面がある」という手作りの
データで、階の分け方と、床・階段・家具など・壁の分け方を確かめる。
"""

import numpy as np
import pytest

import scan_floorplan as fp

M2 = 1.0e4   # 1㎡ を cm² で


def _faces(*heights_and_areas):
    """[(高さcm, 面積㎡), ...] を、find_levels に渡す z と area にする。"""
    z = np.array([h for h, _ in heights_and_areas], dtype=float)
    area = np.array([a * M2 for _, a in heights_and_areas], dtype=float)
    return z, area


# ---- find_levels ----

def test_find_levels_takes_two_wide_heights_as_two_floors():
    # 段(1段0.2㎡)が間に並んでいても、階は床の2つだけ
    steps = [(20.0 * i + 5.0, 0.2) for i in range(1, 13)]
    levels = fp.find_levels(*_faces((5.0, 8.0), (275.0, 6.0), *steps))
    assert levels == pytest.approx([5.0, 275.0])


def test_find_levels_merges_a_small_step_into_one_floor():
    # 玄関の段差(20cm)は同じ階。広いほうの高さを床の高さにする
    levels = fp.find_levels(*_faces((5.0, 1.5), (25.0, 7.0)))
    assert levels == pytest.approx([25.0])


def test_find_levels_does_not_take_furniture_tops_as_a_floor():
    # 2階の床(305cm)より狭い、ベッドや机の上面(355cm、365cm)
    levels = fp.find_levels(*_faces((45.0, 5.5), (305.0, 7.0), (355.0, 1.2), (365.0, 1.3)))
    assert levels == pytest.approx([45.0, 305.0])


def test_find_levels_ignores_a_landing_between_floors():
    # 踊り場(0.9㎡)は階にしない
    levels = fp.find_levels(*_faces((5.0, 6.0), (140.0, 0.9), (275.0, 6.0)))
    assert levels == pytest.approx([5.0, 275.0])


def test_find_levels_keeps_one_floor_for_a_tiny_scan():
    # 1㎡に満たない小さなスキャンでも、一番広い高さを階にする
    levels = fp.find_levels(*_faces((5.0, 0.6), (35.0, 0.2)))
    assert levels == pytest.approx([5.0])


def test_find_levels_returns_nothing_without_faces():
    assert fp.find_levels([], []) == []
