"""scan_picker が立体表示に配るファイルの扱いのテスト。

画面そのもの(ブラウザ側)はここでは確かめない。「配るフォルダの外に出ない」
「判定側と同じメッシュを表示に渡す」を確かめる。
"""

import zipfile

import numpy as np
import pytest
import trimesh

import scan_ingest as si
import scan_picker as sp


@pytest.mark.parametrize("name, expected", [
    ("a.obj", "a.obj"),
    ("/scan/a.obj", "a.obj"),
    ("x/y/a b.jpg", "a b.jpg"),
    ("x\\y\\a.obj", "a.obj"),
])
def test_plain_name_takes_the_last_part(name, expected):
    assert sp.plain_name(name) == expected


@pytest.mark.parametrize("name", [
    "D:x.obj",            # folder / "D:x.obj" は別のドライブを指す
    "/scan/D:x.obj",
    "dir/D:x.obj",
    "a:b.jpg",            # 代替データストリーム
    "..", "x/..", "..\\..", ".", "", "x/",
])
def test_plain_name_refuses_names_that_could_leave_the_folder(name):
    assert sp.plain_name(name) is None


def _obj_bytes():
    box = trimesh.creation.box(extents=(2.0, 0.2, 2.0))
    return box.export(file_type="obj").encode("utf-8")


def test_view_of_scan_unpacks_only_plain_scan_files(tmp_path):
    path = tmp_path / "scan.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Scan 1/scan.obj", _obj_bytes())
        z.writestr("Scan 1/scan.mtl", "newmtl main\n")
        z.writestr("Scan 1/scan.jpg", b"jpg")
        z.writestr("Scan 1/readme.txt", "x")          # 決めた種類でない
        z.writestr("Scan 1/D:evil.jpg", b"x")          # 一時フォルダの外を指す名前
        z.writestr("other/second.obj", _obj_bytes())   # 判定側が読まない2つ目のOBJ
    view = sp.view_of_scan(path, np.eye(4).tolist())
    assert view["file"] == "scan.obj"
    assert sorted(f.name for f in view["dir"].iterdir()) == ["scan.jpg", "scan.mtl", "scan.obj"]


def test_view_of_scan_hands_back_nothing_for_a_ply(tmp_path):
    # PLY は立体表示に元ファイルを出せない。呼び出し側が形だけを出す
    path = tmp_path / "scan.ply"
    trimesh.creation.box(extents=(2.0, 0.2, 2.0)).export(path)
    assert sp.view_of_scan(path, np.eye(4).tolist()) is None


def test_view_of_mesh_writes_a_y_up_copy_that_maps_back(tmp_path):
    # 判定側の座標(上がZ)のメッシュを、上がYで書き出し、行列で元に戻せること
    mesh = trimesh.creation.box(extents=(100.0, 60.0, 20.0))
    mesh.apply_translation((50.0, 30.0, 10.0))
    view = sp.view_of_mesh(mesh)
    shown = si.load_mesh(str(view["dir"] / view["file"]))
    assert np.ptp(shown.vertices[:, 1]) == pytest.approx(20.0)   # 高さがY
    T = np.array(view["to_zup"])
    back = (T @ np.column_stack([shown.vertices, np.ones(len(shown.vertices))]).T).T[:, :3]
    assert back.min(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert back.max(axis=0) == pytest.approx([100.0, 60.0, 20.0], abs=1e-6)
