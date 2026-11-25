"""scan_scene: 判定の結果を立体表示に重ねる形にする所のテスト。

掃引そのものは回さない(時間がかかる)。合成経路の中心線の上に家具を置いた
姿勢を手で2つ作り、それが正しい座標・正しい大きさで出てくることを確かめる。
"""

import numpy as np
import pytest

import mesh_env as M
import phase2_lstair as L
import route_env as R
import scan_candidates as sc
import scan_plan as sp
import scan_scene as scn
import scan_space as ss

DIMS = (100.0, 40.0, 50.0)   # 進行方向の長さ・幅・高さ


@pytest.fixture(scope="module")
def scene_and_matrix():
    mesh = M.env_to_mesh(*R.build_route())
    mesh.invert()
    space = ss.ScanSpace(mesh, (42.5, 42.5, 100.0))
    space.build_centerline((42.5, 42.5, 0.0), (447.5, 462.5, 20.0))
    # 立体表示のファイルの座標は、上がY・単位がm だとする
    to_zup = np.array([[100.0, 0, 0, 5.0], [0, 0, -100.0, 7.0], [0, 100.0, 0, 9.0], [0, 0, 0, 1]])
    with sc.furniture_dims(*DIMS):
        poses = [sp.state_at(space, s) for s in (150.0, 165.0)]
        pose = [(p.tolist(), q.tolist()) for p, q in poses]
        neck = {"s": 165.0, "capacity_cm": -4.0,
                "profile": [[150.0, 12.0], [165.0, -4.0], [180.0, -np.inf]],
                "poses": [pose[0], pose[1], None],
                "contact": {"xyz": [10.0, 20.0, 30.0], "part": "furniture"}}
        result = {"bottleneck": neck, "bottleneck_furniture_only": {"capacity_cm": 6.0}}
        scene = scn.result_scene(space, result, 2, np.linalg.inv(to_zup))
    return scene, to_zup, poses


def _zup(flat, to_zup):
    pts = np.array(flat).reshape(-1, 3)
    return pts @ to_zup[:3, :3].T + to_zup[:3, 3]


def test_each_swept_position_becomes_a_frame(scene_and_matrix):
    scene, _, _ = scene_and_matrix
    assert [f["s"] for f in scene["frames"]] == [150.0, 165.0, 180.0]
    assert scene["frames"][0]["clearance"] == 12.0
    assert scene["bottleneck"]["frame"] == 1
    assert scene["furniture_only_cm"] == 6.0


def test_position_without_a_holdable_pose_has_no_shapes(scene_and_matrix):
    last = scene_and_matrix[0]["frames"][2]
    assert last["clearance"] is None and last["box"] is None and last["people"] is None
    assert len(last["at"]) == 3     # 経路の線は引けるよう、位置だけは持つ


def test_furniture_box_has_the_carried_size_at_the_swept_pose(scene_and_matrix):
    scene, to_zup, poses = scene_and_matrix
    corners = _zup(scene["frames"][0]["box"], to_zup)
    assert corners.shape == (8, 3)
    assert corners.mean(axis=0) == pytest.approx(poses[0][0], abs=0.02)
    # 角0から、3つの軸それぞれの隣の角までの長さが寸法になる
    edges = sorted(np.linalg.norm(corners[i] - corners[0]) for i in (4, 2, 1))
    assert edges == pytest.approx(sorted(DIMS), abs=0.02)
    assert max(scene["box_tris"]) == 7 and len(scene["box_tris"]) == 36


def test_carriers_stand_on_the_floor_beside_the_furniture(scene_and_matrix):
    scene, to_zup, poses = scene_and_matrix
    frame = scene["frames"][0]
    assert len(frame["people"]) == 2
    for flat in frame["people"]:
        body = _zup(flat, to_zup)
        assert max(scene["people_tris"]) == len(body) - 1
        assert body[:, 2].min() == pytest.approx(R.STEP_H, abs=3.0)   # 足は廊下の床
        assert np.ptp(body[:, 2]) == pytest.approx(170.0, abs=8.0)   # 身長
        # 家具の中心から、家具の長さの半分より離れた所に立つ
        dist = np.linalg.norm(body[:, :2].mean(axis=0) - poses[0][0][:2])
        assert dist > DIMS[0] / 2


def test_contact_point_is_given_in_the_file_coordinates(scene_and_matrix):
    scene, to_zup, _ = scene_and_matrix
    assert _zup(scene["bottleneck"]["contact"], to_zup)[0] == pytest.approx([10.0, 20.0, 30.0], abs=0.02)
