"""スキャンの読み込みで上下が正しく立つことのテスト。

2026-11-17 まで、Y-up のスキャンを Z-up に直す回転の向きが逆で、天井が床として
扱われていた。合成データは最初から Z-up で作るのでこの回転を通らず、テストで
見つからなかった。ここでは Y-up で作った「床の上に段が1つ」のメッシュを通す。
"""

import os

import numpy as np
import pytest
import trimesh

import scan_ingest as si

REAL_SCAN = r"C:\dev\alpha-project\Scaniverse 2026-09-19 113017 stair1.zip"


def _yup_floor_with_step(tmp_path):
    """Y が上の座標で、床(2m角)の上に段(高さ0.2m)を置いたOBJを書き出す。"""
    floor = trimesh.creation.box(extents=(2.0, 0.02, 2.0))
    floor.apply_translation((0.0, 0.01, 0.0))
    step = trimesh.creation.box(extents=(0.5, 0.2, 0.5))
    step.apply_translation((0.5, 0.12, 0.5))
    path = tmp_path / "yup.obj"
    trimesh.util.concatenate([floor, step]).export(path)
    return str(path)


def test_yup_scan_stands_upright(tmp_path):
    m, info = si.load_scan_zup(_yup_floor_with_step(tmp_path))
    v = np.asarray(m.vertices)
    # 単位はメートルと推定されて cm になる。高さは 22cm
    assert np.ptp(v[:, 2]) == pytest.approx(22.0, abs=0.5)
    # 段(水平の広がりが小さい側)が上にある
    top = v[v[:, 2] > 15.0]
    bottom = v[v[:, 2] < 1.0]
    assert np.ptp(top[:, 0]) < np.ptp(bottom[:, 0])


def test_upward_share_detects_flipped_mesh():
    # 上向きの面だけを持つ板。裏返すと割合が0になる
    plate = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                            faces=[[0, 1, 2], [0, 2, 3]], process=False)
    assert si.upward_share(plate) == pytest.approx(1.0)
    plate.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    assert si.upward_share(plate) == pytest.approx(0.0)


@pytest.mark.skipif(not os.path.exists(REAL_SCAN), reason="実スキャンはリポジトリ外にある")
def test_real_stair_scan_has_treads_facing_up():
    """自宅の階段のスキャン: 1階の床から2階の床(約270cm)まで、段の上面が上を向く。"""
    m, info = si.load_scan_zup(REAL_SCAN)
    assert info["upward_share"] > 0.5
    z = m.triangles_center[:, 2]
    up = m.face_normals[:, 2] > 0.9
    area = m.area_faces
    # 床から2階までの間、20cm刻みの各帯に上向きの面(踏み面)がある
    for z0 in range(20, 260, 40):
        assert area[up & (z >= z0) & (z < z0 + 40)].sum() > 500.0, z0
