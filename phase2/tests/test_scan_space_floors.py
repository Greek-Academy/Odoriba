"""scan_space の床の検出のテスト。

下の階の天井が撮れていないスキャンでも、上の階の床が床として登録されること。
"""

import numpy as np
import pytest
import trimesh

import scan_space as ss


def _quad(corners, facing):
    """4つの角から、facing の向きを表にした面(三角形2枚)を作る。"""
    c = np.asarray(corners, dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    if np.dot(np.cross(c[1] - c[0], c[2] - c[0]), facing) < 0:
        faces = faces[:, ::-1]
    return c, faces


def _two_storey_without_lower_ceiling():
    """吹き抜けでつながった2階建て。面は撮った人の側(空間の側)を向く。

    1階の床 z=0、2階の床 z=251 は y<120 の範囲だけの1枚の面で、その裏
    (1階の天井)は無い。y>=120 は吹き抜けで、上下の空間がつながっている。
    2階の床の高さは格子(2cm刻み)から外してある。実物のスキャンと同じく、
    面を含むボクセルの中心が面の上にずれ、そのボクセルが自由空間になる
    (格子ちょうどの高さだと、面のボクセルが壁扱いになって不具合が出ない)。
    """
    W, D, H, Z2, Y2 = 200.0, 200.0, 500.0, 251.0, 120.0
    quads = [
        ([(0, 0, 0), (W, 0, 0), (W, D, 0), (0, D, 0)], (0, 0, 1)),        # 1階の床
        ([(0, 0, H), (W, 0, H), (W, D, H), (0, D, H)], (0, 0, -1)),       # 2階の天井
        ([(0, 0, 0), (W, 0, 0), (W, 0, H), (0, 0, H)], (0, 1, 0)),        # 壁
        ([(0, D, 0), (W, D, 0), (W, D, H), (0, D, H)], (0, -1, 0)),
        ([(0, 0, 0), (0, D, 0), (0, D, H), (0, 0, H)], (1, 0, 0)),
        ([(W, 0, 0), (W, D, 0), (W, D, H), (W, 0, H)], (-1, 0, 0)),
        ([(0, 0, Z2), (W, 0, Z2), (W, Y2, Z2), (0, Y2, Z2)], (0, 0, 1)),  # 2階の床(1枚)
    ]
    verts, faces = [], []
    for corners, facing in quads:
        v, f = _quad(corners, np.array(facing, dtype=float))
        faces.append(f + 4 * len(verts))
        verts.append(v)
    return trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)


@pytest.fixture(scope="module")
def space():
    return ss.ScanSpace(_two_storey_without_lower_ceiling(), (100.0, 160.0, 100.0))


def test_upper_floor_is_found_where_the_ceiling_below_was_not_scanned(space):
    # 2階の床の裏まで1階の空間が届いていても、2階の床は床になる
    assert space.floor_fn(100.0, 60.0, z_hint=350.0) == pytest.approx(251.0, abs=3.0)


def test_lower_floor_is_still_found_under_the_upper_floor(space):
    assert space.floor_fn(100.0, 60.0, z_hint=100.0) == pytest.approx(0.0, abs=3.0)
    assert space.floor_fn(100.0, 160.0, z_hint=100.0) == pytest.approx(0.0, abs=3.0)


def test_the_void_has_no_upper_floor(space):
    # 吹き抜けには2階の床が無い(高い所から見ても、返るのは1階の床)
    assert space.floor_fn(100.0, 170.0, z_hint=350.0) == pytest.approx(0.0, abs=3.0)
