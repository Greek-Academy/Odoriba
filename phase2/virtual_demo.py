"""
数値から仮想階段を組んで、家具を通すデモ(スキャンの見た目に頼らない版)。

階段の寸法(幅・蹴上げ・踏み面・段数・踊り場)を数値で指定して仮想の
L字階段を作り、家具(寸法を数値で指定、またはスキャンから)を通せるか
判定して、きれいな3D(HTML)とGIFを出す。

例:
  # 数値だけで: 幅80cm・蹴上げ18・踏み面25・各10段の階段に、幅60x高さ40x長さ180の家具を2人で
  python virtual_demo.py --width 80 --rise 18 --tread 25 --steps 10 \\
      --furniture 180 60 40 --carriers 2 --html out.html

  # 家具はスキャンから、階段は数値で
  python virtual_demo.py --width 78 --rise 18 --tread 25 --steps 13 \\
      --furniture-scan box.obj --html out.html
"""

import argparse

import numpy as np

import geometry3d as g3
import phase2_lstair as L
import scan_furniture as sf
import measure_stairs as ms


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=float, help="階段・踊り場の幅(cm)")
    ap.add_argument("--rise", type=float, help="蹴上げ=1段の高さ(cm)")
    ap.add_argument("--tread", type=float, help="踏み面=1段の奥行き(cm)")
    ap.add_argument("--steps", type=int, help="1フライトの段数(上下とも同数にする)")
    ap.add_argument("--n1", type=int, help="下フライトの段数(--stepsの代わりに個別指定)")
    ap.add_argument("--n2", type=int, help="上フライトの段数")
    ap.add_argument("--ceil", type=float, help="天井高(最上段からの高さ, cm)")
    ap.add_argument("--measure-stair", metavar="SCAN.obj",
                    help="スキャンした階段から寸法(蹴上げ・踏み面・幅・段数)を計測して"
                         "仮想階段に使う。個別の--width等を併記すればそちらで上書き")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--furniture", type=float, nargs=3, metavar=("L", "W", "H"),
                   help="家具の寸法(長さ 幅 高さ, cm)")
    g.add_argument("--furniture-scan", help="家具のスキャンOBJ/PLY(寸法を読み取る)")
    ap.add_argument("--carriers", type=int, choices=(0, 1, 2), default=0)
    ap.add_argument("--max-iter", type=int, default=4000)
    ap.add_argument("--html", help="3Dビューア(HTML)の出力先")
    ap.add_argument("--gif", help="アニメGIFの出力先")
    args = ap.parse_args()

    # --- 階段寸法: スキャン計測を土台に、明示指定があればそれで上書き ---
    width, rise, tread = args.width, args.rise, args.tread
    n1 = args.n1 if args.n1 is not None else args.steps
    n2 = args.n2 if args.n2 is not None else args.steps
    if args.measure_stair:
        r = ms.measure(args.measure_stair)
        print("スキャン計測: 蹴上げ%.0f / 踏み面%.0f(粗) / 幅%.0f(粗) / 段数%d(各%d)"
              % (r["rise"], r["tread"], r["width"], r["n_steps_total"],
                 r["n_steps_total"] // 2))
        # 明示指定が無い項目だけ計測値で埋める
        if width is None: width = r["width"]
        if rise is None: rise = r["rise"]
        if tread is None: tread = r["tread"]
        half = max(1, r["n_steps_total"] // 2)
        if n1 is None: n1 = half
        if n2 is None: n2 = half
    ceil = args.ceil
    if ceil is None and args.measure_stair:
        ceil = r["headroom"]                        # 天井も計測値を使う
        print("スキャン計測: 頭上クリアランス %.0fcm" % ceil)
    L.configure(width=width, rise=rise, tread=tread,
                n_steps1=n1, n_steps2=n2, ceil_clear=ceil)
    print(f"仮想階段: 幅{L.STAIR_WIDTH:.0f}cm / 蹴上げ{L.RISE:.0f} x 踏み面{L.TREAD:.0f}cm / "
          f"({L.N_STEPS1}+{L.N_STEPS2})段 / 踊り場{L.STAIR_WIDTH:.0f}角 / "
          f"全高{L.TOP_Z:.0f}cm")

    # --- 家具(寸法 or スキャン) ---
    if args.furniture:
        dims = np.array(sorted(args.furniture, reverse=True), dtype=float)
        # 直方体の表面サンプル点を家具ローカル点として用意(長辺=x)
        local = g3.box_surface_sample_points(dims / 2, nu=8, nv=4, nw=4)
        furn = {"local": local, "dims": dims}
    else:
        pts = sf.isolate_object_points(args.furniture_scan)
        furn = sf.load_furniture(points=pts)
    print(f"家具: {furn['dims'].round(0)} cm (長辺x中間x短辺)")

    # --- 判定 ---
    res = sf.judge_in_lstair(furn, max_iter=args.max_iter, num_carriers=args.carriers)
    who = "家具単体" if args.carriers == 0 else f"運搬者{args.carriers}人"
    verdict = "PASS(通る)" if res["found"] else "BLOCKED(この試行では見つからず)"
    print(f"{who}: {verdict}")

    # --- 可視化(通れば経路、詰まればbest-effort=最遠到達を赤で) ---
    if args.html or args.gif:
        vpath = res["path"] if res["found"] else res.get("best_effort")
        if vpath is None:
            print("可視化用の経路が得られませんでした")
        else:
            if args.gif:
                sf.render_gif(furn, vpath, args.gif)
            if args.html:
                sf.render_3d(furn, vpath, args.html, num_carriers=args.carriers,
                             stuck=not res["found"])
