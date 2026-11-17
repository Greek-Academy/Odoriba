"""スキャンからの階段寸法の計測(段数・全高・蹴上げ)のテスト。

2026-11-17 まで、天井(下向きの面)まで段として数えていて、13段の階段が
27段・全高520cmと出ていた。上向きの面だけを使い、下の階の床から上の階の床
までを数える。
"""

import os

import numpy as np
import pytest
import trimesh

import measure_stairs as ms

REAL_SCAN = r"C:\dev\alpha-project\Scaniverse 2026-09-19 113017 stair1.zip"


def _yup_straight_stair(tmp_path, n_treads=5, rise=0.18, tread=0.25, width=0.8):
    """Y が上の座標(メートル)で、床・まっすぐな階段・上の階の床・天井を作る。"""
    parts = []

    def box(x0, x1, z0, z1, top):
        b = trimesh.creation.box(extents=(x1 - x0, top, z1 - z0))
        b.apply_translation(((x0 + x1) / 2, top / 2, (z0 + z1) / 2))
        parts.append(b)

    floor = trimesh.creation.box(extents=(3.0, 0.02, 2.0))
    floor.apply_translation((0.5, -0.01, 0.4))
    parts.append(floor)
    for k in range(n_treads):
        box(k * tread, (k + 1) * tread, 0.0, width, (k + 1) * rise)
    top = (n_treads + 1) * rise
    box(n_treads * tread, n_treads * tread + 1.5, 0.0, width, top)   # 上の階の床
    ceiling = trimesh.creation.box(extents=(3.0, 0.02, 2.0))         # 天井は数えない
    ceiling.apply_translation((0.5, top + 2.4, 0.4))
    parts.append(ceiling)
    mesh = trimesh.util.concatenate(parts)
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.1)
    path = tmp_path / "stair.obj"
    trimesh.Trimesh(v, f, process=False).export(path)
    return str(path)


def test_counts_rises_between_floors(tmp_path):
    r = ms.measure(_yup_straight_stair(tmp_path))
    assert r["n_steps_total"] == 6                      # 踏み面5枚 = 蹴上げ6つ
    assert r["total_height"] == pytest.approx(108.0, abs=2.0)
    assert r["rise"] == pytest.approx(18.0, abs=0.5)


@pytest.mark.skipif(not os.path.exists(REAL_SCAN), reason="実スキャンはリポジトリ外にある")
def test_real_stair_scan_matches_tape_measure():
    """自宅の階段: メジャーでは13段・蹴上げ約21cm(2026-11-17)。"""
    r = ms.measure(REAL_SCAN)
    assert r["n_steps_total"] == 13
    assert r["rise"] == pytest.approx(21.0, abs=1.0)
    assert r["total_height"] == pytest.approx(270.0, abs=8.0)
