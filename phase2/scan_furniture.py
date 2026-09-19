"""
スキャンした家具(段ボール等)を「運ぶ剛体」として判定に使う。

これまで家具は直方体(FURN_L x W x H)の近似だったが、ここではスキャンした
メッシュの表面点をそのまま家具の形として使い、L字階段(phase2_lstair)を
通れるか(通る/通らない・最小クリアランス)を判定する。段ボールのような
閉じた物体はスキャンが水密化しやすく、家具スキャンの実証に向く。

パイプライン:
  scan_ingestで単位をcmに正規化 -> OBB(有向境界箱)で「長辺=ローカルx」に
  向きを揃えて中心を原点へ -> 表面点をサンプル -> その点群を家具として
  L字階段のクリアランス評価・RRT探索にかける。

いまの範囲: 家具単体(運搬者なし)の通過判定。運搬者ありは把持点の定義を
スキャン家具に合わせる必要があり、次段階。

実行:
    python scan_furniture.py <furniture.obj>              # L字階段を通れるか
    python scan_furniture.py <furniture.obj> --max-iter 2000
    python scan_furniture.py --selftest                  # 合成の箱で自己テスト
"""

import argparse

import numpy as np
import trimesh

import phase2_demo as p
import phase2_lstair as L
import scan_ingest as si


def load_furniture(path=None, mesh=None, assume_units=None, n_points=250):
    """家具スキャンを読み、OBBで姿勢を正規化して表面点(ローカル)を返す。

    戻り値: dict(local=表面点(N,3,中心原点・長辺がx), dims=(L,W,H降順), mesh=正規化後)
    pathの代わりにmeshを直接渡してもよい(自己テスト用)。
    """
    if mesh is None:
        mesh = si.load_mesh(path)
        scale, _ = si.guess_units_scale(mesh, assume_units)
        mesh.apply_scale(scale)

    # 有向境界箱(OBB)で、物体の主軸をワールド軸に揃える
    obb = mesh.bounding_box_oriented
    Tinv = np.linalg.inv(obb.primitive.transform)
    mesh = mesh.copy()
    mesh.apply_transform(Tinv)  # OBB中心が原点、OBB軸が座標軸

    ext = np.asarray(mesh.extents, dtype=float)
    order = np.argsort(ext)[::-1]  # 長い順 -> x=長辺, y=中間, z=短辺
    dims = ext[order]

    # 表面点をサンプル(なるべく一様に)。少数の頂点だけだと面の中央を
    # 拾い損ねるので、mesh.sampleで面上から取る。
    try:
        pts = mesh.sample(n_points)
    except Exception:
        pts = np.asarray(mesh.vertices)[:n_points]
    local = np.asarray(pts)[:, order]  # 軸を長い順に並べ替え
    return {"local": local, "dims": dims, "mesh": mesh}


def furniture_world_points(local, pos, quat):
    """ローカル表面点を姿勢(pos,quat)でワールド座標へ。
    phase2_lstair.furniture_pointsのスキャン家具版。"""
    R = p.g3.rotmat_from_quat(quat)
    return (np.asarray(local) @ R.T) + np.asarray(pos)


def _start_goal(h):
    """家具高さhに合わせたSTART/GOAL(catalog.pyと同じ組み立て)。"""
    start = (np.array([L.CENTER, 170.0, h / 2]), L._pose(90.0, 0.0).as_quat())
    goal = (np.array([L.FL2_X1 + 170.0, L.LAND_YC, L.TOP_Z + h / 2]),
            L._pose(0.0, 0.0).as_quat())
    return start, goal


def judge_in_lstair(furn, max_iter=3000, seeds=(0, 1)):
    """スキャン家具がL字階段を通れるか(家具単体)。

    家具の外形寸法(dims)をL.FURN_*に入れてサンプラー・START/GOALを
    整えつつ、クリアランス判定だけはスキャン点群で行う(validatorを差し替え)。
    戻り値: dict(found, min_clearance, dims)
    """
    local, dims = furn["local"], furn["dims"]
    orig = (L.FURN_L, L.FURN_W, L.FURN_H)
    L.FURN_L, L.FURN_W, L.FURN_H = map(float, dims)
    try:
        outer, obstacles = L.build_lstairs()
        start, goal = _start_goal(dims[2])

        def validator(pos, quat, with_human, outer, obstacles, num_carriers=None):
            # 家具単体のみ(with_humanは無視)。スキャン点群で最小クリアランス
            return p.shape_clearance_3d(
                furniture_world_points(local, pos, quat), outer, obstacles) >= 0

        found = False
        min_clear = None
        for sd in seeds:
            path = p.rrt_connect(
                start, goal, with_human=False, outer=outer, obstacles=obstacles,
                max_iter=max_iter, seed=sd, w_rot=L.W_ROT,
                sampler=L.make_sampler(with_human=False), validator=validator)
            if path is not None:
                found = True
                min_clear = min(
                    p.shape_clearance_3d(furniture_world_points(local, ps, q),
                                         outer, obstacles)
                    for ps, q in path)
                break
        return {"found": found, "min_clearance": min_clear, "dims": dims}
    finally:
        L.FURN_L, L.FURN_W, L.FURN_H = orig


def _selftest():
    """合成の箱(段ボール想定)で、寸法の取り出しと通過判定が動くか。"""
    print("=== 自己テスト: 合成の箱を家具スキャンに見立てる ===")
    for name, ext in [("小さい箱 40x30x30", [40, 30, 30]),
                      ("大きい箱 250x60x60", [250, 60, 60])]:
        box = trimesh.creation.box(extents=ext)
        furn = load_furniture(mesh=box)
        print(f"{name}: 取り出したdims(cm)= {furn['dims'].round(0)}")
        res = judge_in_lstair(furn, max_iter=1500)
        verdict = "PASS" if res["found"] else "BLOCKED"
        mc = "" if res["min_clearance"] is None else f" (min clearance {res['min_clearance']:.1f}cm)"
        print(f"  L字階段: {verdict}{mc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("furniture", nargs="?", help="家具の OBJ/PLY")
    parser.add_argument("--assume-units", choices=("m", "cm", "mm"))
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    elif args.furniture:
        furn = load_furniture(args.furniture, assume_units=args.assume_units)
        print(f"家具スキャン: dims(cm)= {furn['dims'].round(0)} (長辺x中間x短辺)")
        res = judge_in_lstair(furn, max_iter=args.max_iter)
        verdict = "PASS(通る)" if res["found"] else "BLOCKED(この試行では見つからず)"
        mc = "" if res["min_clearance"] is None else f" / 最小クリアランス {res['min_clearance']:.1f}cm"
        print(f"L字階段(幅{L.STAIR_WIDTH:.0f}cm・踊り場{L.STAIR_WIDTH:.0f}角): {verdict}{mc}")
    else:
        parser.error("furniture を指定するか --selftest を使ってください")
