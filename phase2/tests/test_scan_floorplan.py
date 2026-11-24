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


# ---- plans_from_points ----

def _patch(x0, x1, y0, y1, z, step=2.5):
    """水平な長方形の面の上の点(上向きの面の代わり)。"""
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, float(z))])


def _wall(x, y0, y1, z0, z1, step=2.5):
    """x が一定の、縦の面の上の点。"""
    ys = np.arange(y0 + step / 2, y1, step)
    zs = np.arange(z0 + step / 2, z1, step)
    gy, gz = np.meshgrid(ys, zs)
    return np.column_stack([np.full(gy.size, float(x)), gy.ravel(), gz.ravel()])


@pytest.fixture
def house():
    """1階(z=0、x 0〜200)と2階(z=270、x 300〜400)を、10段の階段でつないだ家。

    1階に机(z=70)と壁(x=2)、低い縦の面(x=150、高さ20cm)がある。
    """
    steps = [_patch(200 + 10 * i, 210 + 10 * i, 0, 100, 27 * (i + 1)) for i in range(9)]
    up = np.vstack([_patch(0, 200, 0, 100, 0), _patch(300, 400, 0, 100, 270),
                    _patch(50, 100, 20, 60, 70), *steps])
    side = np.vstack([_wall(2, 0, 100, 0, 240), _wall(150, 0, 100, 0, 20)])
    return fp.plans_from_points(up, side, [0.0, 270.0], [[0, 0], [400, 100]])


def _at(plans, floor, x, y):
    ix = int((x - plans["x0"]) // plans["cell"])
    iy = int((y - plans["y0"]) // plans["cell"])
    f = plans["floors"][floor]
    return int(f["kind"][iy, ix]), float(f["z"][iy, ix])


def test_plans_split_the_house_into_two_named_floors(house):
    assert [f["name"] for f in house["floors"]] == ["1階", "2階"]
    assert house["floors"][0]["kind"].shape == (house["ny"], house["nx"]) == (20, 80)


def test_plans_show_each_floor_only_on_its_own_plan(house):
    assert _at(house, 0, 120, 80) == (fp.FLOOR, 0.0)
    assert _at(house, 1, 120, 80)[0] == fp.NOTHING
    assert _at(house, 1, 350, 50) == (fp.FLOOR, 270.0)
    assert _at(house, 0, 350, 50)[0] == fp.NOTHING


def test_plans_show_the_stairs_on_both_floors_with_their_height(house):
    # x=252 は6段目(z=162)
    for floor in (0, 1):
        kind, z = _at(house, floor, 252, 50)
        assert kind == fp.STAIR
        assert z == pytest.approx(162.0)


def test_plans_mark_a_table_top_as_not_placeable(house):
    kind, z = _at(house, 0, 75, 40)
    assert kind == fp.OTHER
    assert np.isnan(z)


def test_plans_draw_a_tall_vertical_face_as_a_wall_but_not_a_low_one(house):
    kind, z = _at(house, 0, 2, 50)
    assert kind == fp.WALL and np.isnan(z)
    assert _at(house, 0, 150, 50)[0] == fp.FLOOR   # 高さ20cmの縦の面は壁にしない


def test_plans_only_give_a_height_to_floor_and_stair_cells(house):
    for f in house["floors"]:
        placeable = (f["kind"] == fp.FLOOR) | (f["kind"] == fp.STAIR)
        assert not np.isnan(f["z"][placeable]).any()
        assert np.isnan(f["z"][~placeable]).all()


def test_plans_keep_a_half_scanned_stair_off_the_floor_it_does_not_reach():
    # 1階から4段だけ撮れた階段(2階の床にはつながらない)
    steps = [_patch(200 + 10 * i, 210 + 10 * i, 0, 100, 35 + 27 * i) for i in range(4)]
    up = np.vstack([_patch(0, 200, 0, 100, 0), _patch(300, 400, 0, 100, 270), *steps])
    plans = fp.plans_from_points(up, np.zeros((0, 3)), [0.0, 270.0], [[0, 0], [400, 100]])
    assert _at(plans, 0, 222, 50)[0] == fp.STAIR
    assert _at(plans, 1, 222, 50)[0] == fp.NOTHING


def test_to_json_turns_cells_without_a_height_into_none(house):
    data = fp.to_json(house)
    assert data["floors"][0]["z"][10][24] == 0.0          # 床
    assert data["floors"][0]["z"][8][15] is None          # 机の上
    assert data["floors"][1]["kind"][10][70] == fp.FLOOR
    import json
    json.dumps(data)   # そのまま送れる
