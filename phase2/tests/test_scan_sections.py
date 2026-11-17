"""スキャンから経路に沿って幅と天井を測る計測のテスト。

寸法の分かっている廊下(幅80cm・天井230cm、片側に手すりの出っ張り7cm)を
作り、壁から壁の幅・出っ張り込みの幅・天井が2cm程度で読めることを確かめる。
実スキャンでは、メジャーで測った直線部分の幅75cmと5cm程度で合うことを見る。
"""

import os

import numpy as np
import pytest
import trimesh

import scan_sections as sc

REAL_SCAN = r"C:\dev\alpha-project\Scaniverse 2026-09-19 113017 stair1.zip"


def _corridor(width=80.0, ceiling=230.0, length=300.0, rail=7.0):
    """X 方向に伸びる廊下。面の向きが廊下の内側を向くよう、板を外側に置く。"""
    def box(x0, x1, y0, y1, z0, z1):
        b = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
        b.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
        return b

    return trimesh.util.concatenate([
        box(0, length, -10, width + 10, -5, 0),                  # 床
        box(0, length, -10, width + 10, ceiling, ceiling + 5),   # 天井
        box(0, length, -10, 0, 0, ceiling),                      # 壁
        box(0, length, width, width + 10, 0, ceiling),           # 壁
        box(0, length, 0, rail, 80, 86),                         # 手すり(高さ80〜86cm)
    ])


def test_corridor_width_and_ceiling():
    mesh = _corridor()
    path = np.array([[x, 40.0, 0.0] for x in np.arange(100.0, 201.0, 25.0)])
    prof = sc.section_profile(mesh, path)
    assert len(prof) == len(path)
    for p in prof:
        assert abs(p["width_wall"] - 80.0) < 2.0     # 壁から壁
        assert abs(p["width_min"] - 73.0) < 2.5      # 手すりの内側
        assert 75.0 <= p["width_min_h"] <= 90.0      # 一番狭いのは手すりの高さ
        assert abs(p["ceiling"] - 230.0) < 2.0
    assert len(sc.by_step(prof)) == 1                # 平らな床は1つにまとまる


def test_width_does_not_depend_on_corridor_direction():
    """廊下が斜めを向いていても同じ幅が出る(向きを決め打ちしていない)。"""
    mesh = _corridor()
    rot = trimesh.transformations.rotation_matrix(np.deg2rad(37.0), [0, 0, 1])
    mesh.apply_transform(rot)
    p0 = trimesh.transformations.transform_points([[150.0, 40.0, 0.0]], rot)
    prof = sc.section_profile(mesh, p0)
    assert abs(prof[0]["width_wall"] - 80.0) < 2.5


@pytest.mark.skipif(not os.path.exists(REAL_SCAN), reason="実スキャンが無い")
def test_real_stair_straight_run_matches_tape():
    """自宅の階段の直線部分(4〜6段目)。メジャーでは幅75cm。"""
    import scan_ingest as si
    mesh, _ = si.load_scan_zup(REAL_SCAN)
    path = np.array([[247.0, 329.0, 89.0], [223.0, 325.0, 110.0], [198.0, 322.0, 131.0]])
    prof = sc.section_profile(mesh, path)
    assert len(prof) == 3
    for p in prof:
        assert abs(p["width_wall"] - 75.0) < 6.0
        assert p["width_min"] < p["width_wall"]      # 手すりの分だけ狭い
        assert p["ceiling"] > 200.0                  # 頭上は十分に高い
