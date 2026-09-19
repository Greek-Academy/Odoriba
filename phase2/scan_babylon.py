"""
Babylon.js版の組み合わせビューア: スキャンした階段をテクスチャ付きで
半透明表示し、その中を家具(箱)が通り抜ける様子をブラウザで見せる。

plotlyの点群/面より見た目が良い(実テクスチャ+半透明で中が見える)。
階段はGLB(テクスチャ埋め込み)にしてHTMLにbase64で同梱するので、
ファイル1つでオフラインでも開ける(ネット接続はBabylon.jsのCDN読み込みに
だけ必要)。

    python scan_babylon.py <stair.obj> <box.obj>
    python scan_babylon.py <stair.obj> <box.obj> --alpha 0.35 --out x.html
"""

import argparse
import base64
import json

import numpy as np
import trimesh

import scan_demo as sd
import scan_furniture as sf
import scan_combined as sc


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Odoriba - scanned staircase (Babylon.js)</title>
<script src="https://cdn.babylonjs.com/babylon.js"></script>
<script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"></script>
<style>html,body{{margin:0;height:100%;overflow:hidden;background:#0e1116}}
#c{{width:100%;height:100%;touch-action:none}}
#cap{{position:absolute;top:8px;left:12px;color:#ddd;font:14px sans-serif}}</style>
</head><body>
<div id="cap">Scanned staircase (semi-transparent) + furniture {dimlabel} — drag to rotate</div>
<canvas id="c"></canvas>
<script>
const GLB = "data:model/gltf-binary;base64,{glb_b64}";
const POSES = {poses_json};
const DIMS = {dims_json};
const TARGET = {target_json};
const RADIUS = {radius};
const ALPHA = {alpha};

const canvas = document.getElementById('c');
const engine = new BABYLON.Engine(canvas, true, {{antialias:true, adaptToDeviceRatio:true}});
const scene = new BABYLON.Scene(engine);
scene.useRightHandedSystem = true;                 // numpy(右手系)と揃える
scene.clearColor = new BABYLON.Color4(0.09,0.10,0.12,1);

// サンドボックス相当の質感: 環境光(IBL)+ACESトーンマッピング。
// PBRマテリアル(glTF由来)がこの環境で綺麗に陰影づけされる。
scene.environmentTexture = BABYLON.CubeTexture.CreateFromPrefilteredData(
  "https://assets.babylonjs.com/environments/environmentSpecular.env", scene);
scene.environmentIntensity = 1.15;
scene.imageProcessingConfiguration.toneMappingEnabled = true;
scene.imageProcessingConfiguration.toneMappingType =
  BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
scene.imageProcessingConfiguration.exposure = 1.15;
scene.imageProcessingConfiguration.contrast = 1.1;

const target = new BABYLON.Vector3(TARGET[0],TARGET[1],TARGET[2]);
const cam = new BABYLON.ArcRotateCamera('cam', -1.4, 1.15, RADIUS*2.2, target, scene);
cam.upVector = new BABYLON.Vector3(0,0,1);          // Z-up
cam.wheelPrecision = 0.4; cam.minZ = 1; cam.attachControl(canvas, true);
cam.useAutoRotationBehavior = false;
// IBLだけだと影が単調なので、弱い方向光を1つ足して立体感を出す
const dl = new BABYLON.DirectionalLight('d', new BABYLON.Vector3(0.4,0.5,-1), scene);
dl.intensity = 0.6;

// 「手前の壁だけ」を消して中を見せる = クリッピング平面。
// カメラと中心の間にある手前側の面を切り取る。回転すると平面も追従して、
// 常にこちら側の壁だけが消える(全体は不透明のまま)。CUTは切り込みの深さ。
const CUT = RADIUS * {cut};
scene.onBeforeRenderObservable.add(() => {{
  const dir = target.subtract(cam.position); dir.normalize();   // カメラ->中心(奥向き)
  const pt = target.subtract(dir.scale(CUT));                    // 中心より手前の点
  // 法線を奥向き(dir)にして、手前側(カメラ〜pt)だけを切り取り、奥+中身を残す
  scene.clipPlane = BABYLON.Plane.FromPositionAndNormal(pt, dir);
}});

// 家具(箱): 明るいオレンジのソリッド。長辺x/中間y/短辺z
const box = BABYLON.MeshBuilder.CreateBox('box',
    {{width:DIMS[0], height:DIMS[1], depth:DIMS[2]}}, scene);
const bm = new BABYLON.PBRMaterial('bm', scene);   // 環境光になじむPBRの箱
bm.albedoColor = new BABYLON.Color3(0.95,0.42,0.08);
bm.metallic = 0.0; bm.roughness = 0.55;
bm.emissiveColor = new BABYLON.Color3(0.25,0.10,0.0);  // 少しだけ自発光で視認性
box.material = bm; box.rotationQuaternion = new BABYLON.Quaternion();

BABYLON.SceneLoader.AppendAsync("", GLB, scene).then(() => {{
  // 階段は不透明のまま。両面を描いて、切り取った断面の内側も見えるように。
  scene.meshes.forEach(m => {{
    if (m === box) return;
    if (m.material) {{
      m.material.alpha = ALPHA;                       // 既定1.0(不透明)
      m.material.backFaceCulling = false;            // 内側の面も描く
    }}
  }});
}});

// 経路に沿って家具を滑らかに動かす(位置は線形、姿勢はSlerp)
let t = 0;
scene.onBeforeRenderObservable.add(() => {{
  t = (t + engine.getDeltaTime()/1000 * 0.5) % POSES.length;
  const i = Math.floor(t), f = t - i, j = (i+1) % POSES.length;
  const a = POSES[i], b = POSES[j];
  box.position.set(
    a.p[0]+(b.p[0]-a.p[0])*f, a.p[1]+(b.p[1]-a.p[1])*f, a.p[2]+(b.p[2]-a.p[2])*f);
  const qa = new BABYLON.Quaternion(a.q[0],a.q[1],a.q[2],a.q[3]);
  const qb = new BABYLON.Quaternion(b.q[0],b.q[1],b.q[2],b.q[3]);
  BABYLON.Quaternion.SlerpToRef(qa, qb, f, box.rotationQuaternion);
}});

engine.runRenderLoop(() => scene.render());
window.addEventListener('resize', () => engine.resize());
</script></body></html>
"""


def build(stair_path, box_path, out_path, alpha=1.0, cut=0.15, n=40):
    stair_mesh, _ = sd.load_scan_zup(stair_path)
    glb = stair_mesh.export(file_type="glb")
    glb_b64 = base64.b64encode(glb).decode("ascii")

    box_mesh, _ = sd.load_scan_zup(box_path)
    box_pts, _ = sc.isolate_box_colored(box_mesh)
    _, ext = trimesh.bounds.oriented_bounds(box_pts)
    dims = np.sort(ext)[::-1]

    poses = sc.box_poses_along_stair(stair_mesh, dims, n=n)
    poses_json = json.dumps([{"p": [float(x) for x in pos],
                              "q": [float(x) for x in quat]} for pos, quat in poses])

    c = stair_mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(stair_mesh.extents) / 2)
    html = _HTML.format(
        glb_b64=glb_b64, poses_json=poses_json,
        dims_json=json.dumps([float(x) for x in dims]),
        target_json=json.dumps([float(x) for x in c]),
        radius=radius, alpha=alpha, cut=cut,
        dimlabel=f"{dims[0]:.0f}x{dims[1]:.0f}x{dims[2]:.0f}cm")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"saved Babylon viewer to {out_path} "
          f"(stair {len(stair_mesh.faces)} faces, box {dims.round(0)}cm, alpha={alpha})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stair")
    parser.add_argument("box")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="階段の不透明度(既定1.0=不透明。手前は切り取りで見せる)")
    parser.add_argument("--cut", type=float, default=0.15,
                        help="手前の切り取り深さ(モデル半径に対する比。大きいほど深く切る)")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--out", default="scan_babylon.html")
    args = parser.parse_args()
    build(args.stair, args.box, args.out, alpha=args.alpha, cut=args.cut, n=args.n)
