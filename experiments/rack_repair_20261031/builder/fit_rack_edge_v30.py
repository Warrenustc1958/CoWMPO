# Copyright 2026 The RLinf Authors.
"""Rack -y outer-edge repair v30: v29 cand10 (v23 cand03 + flat -y half-top plate dz .03, xf 0, ye -.10 W) with
the plate's -y edge extended outwards by ext and an optional raised lip along that outer edge (height lip above the
plate top). Everything else unchanged.
Why: v28 acceptance REJECTED v29 cand10 5/6; the only failure is x+050_y-060 (-y outer edge, model +4 steps). The
y-060 probes slide off the -y outer edge (diag_rack_ym060_v28), so the edge position/height is what is still wrong.
Family design informed by those outcomes (recorded in freeze.json); selection uses public RGB only: static IoU + v12
9-probe regression here, 27-probe dense score in score_rack_dense_v30.py. Acceptance on new unused -y edge positions.
ext = 0, lip = 0, dz = .03 must reproduce v29 cand10 exactly (checked in main).
usage: python fit_rack_edge_v30.py [--index i]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import fit_rack_panel_v23 as v23
from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
BASE = R / "rack_scale_scan_dev_v8/scale00"
PREV = R / "rack_halftop_dev_v29/cand10"
OUT = R / "rack_edge_dev_v30"
V12_FREEZE = R / "rack_back_dev_v12/freeze.json"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
DZ0, X00, YE0 = .005, 0., .35                     # v23 cand03 rear top panel
XF, YE = 0., -.10                                 # v29 cand10 half-top plate (front edge x, +y end fraction)
# (half-top plate height above H (m), outward extension of its -y edge (m), lip height above the plate top (m))
GRID = [(dz, ext, lip) for dz in (.03, .035) for ext in (0., .01, .02) for lip in (0., .01, .02)]
RGBA = '0.45 0.3 0.2 1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out_dir, width, depth, height, incline, dz, ext, lip):
    """v23 build_post at cand03 values plus the v29 half-top plate, with its -y edge extended and an optional lip."""
    orig = v23._wrap

    def wrap(name, assets, body, top, radius):
        xb, yl, yr, zt = depth / 2, -width / 2 - ext, YE * width, height + dz
        body += v23._part('box', [(xb - XF) / 2, (yr - yl) / 2, .004], [(xb + XF) / 2, (yr + yl) / 2, zt - .004],
                          RGBA, density=600., friction=.7)
        if lip > 0:
            body += v23._part('box', [(xb - XF) / 2, .004, lip / 2], [(xb + XF) / 2, yl + .004, zt + lip / 2],
                              RGBA, density=600., friction=.7)
        return orig(name, assets, body, max(top, zt + lip), radius + ext)
    v23._wrap = wrap
    try:
        return v23.build_post(out_dir, width, depth, height, incline, DZ0, X00, YE0)
    finally:
        v23._wrap = orig


def worker(i):
    dz, ext, lip = GRID[i]
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    inc = base["incline"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(base["world"], world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(post_raise_neg=v23.H_N, post_raise_pos=v23.HP, post_bar=v23.BAR, post_y_neg=v23.PN,
                       bar_y_neg=v23.YN, bar_end_pos=v23.PE, post_y_pos=v23.PY, panel_dz=DZ0, panel_x0=X00,
                       panel_y_end=YE0, halftop_dz=float(dz), halftop_xf=XF, halftop_ye=YE,
                       edge_ext=float(ext), edge_lip=float(lip))
    ob["top_z"] = float(H + max(v23.H_N, DZ0, dz + lip))
    build(world / "assets/rack", W, D, H, inc, dz, ext, lip)
    meta.pop("world_sha256")
    meta["rack_edge_v30"] = {"base": str(BASE), "prev": str(PREV), "dz": dz, "ext": ext, "lip": lip,
                             "rule": "v29 cand10 half-top plate, -y edge extended by ext + lip; rest unchanged"}
    meta["world_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    (world / "world.json").write_text(json.dumps(meta, indent=1))
    states = Path(base["states"])
    wh = tree_sha(world)
    scene = Scene(str(world))
    docs = {ep: json.loads((states / ep / "state.json").read_text()) for ep in EPS}
    static = []
    for ep in EPS:
        robot = json.loads((R / "robot_rgb_all_dev_v2" / ep / "robot.json").read_text())["robot"]
        scene.reset()
        apply_state(scene, docs[ep]["objects"], require_matched=True, robot=robot)
        k = scene.names.index("rack")
        with np.load(R / "visual_dev_bound_v1/dev3" / ep / "masks.npz") as a:
            target = a["rack"]
        static.append(iou(scene.seg() == k, target))
    robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    responses = []
    for q in json.loads((OUT / "freeze.json").read_text())["queries"]:
        scene.reset()
        apply_state(scene, docs[EP]["objects"], require_matched=True, robot=robot0)
        probe = json.loads(Path(q["source"]).read_text())
        vals = []
        for step, act in enumerate(probe["actions"], 1):
            scene.env.step(np.asarray(act))
            if step not in q["frames"]:
                continue
            fr = Path(q["video"]) / f"{EP}__f{step:03d}"
            m = json.loads((fr / "association.json").read_text())
            if m["masks_sha256"] != sha(fr / "masks.npz") or m["objects"]["robot"]["status"] != "matched":
                raise ValueError(f"invalid response evidence {fr}")
            with np.load(fr / "masks.npz") as a:
                vals.append(iou(scene.robot_mask(), a["robot"]))
            if step >= max(q["frames"]):
                break
        if len(vals) != len(q["frames"]):
            raise ValueError("incomplete contact response")
        responses.append({"key": q["key"], "group": q["group"], "mean_iou": float(np.mean(vals))})
    scene.env.close()
    verify_tree(world, wh)
    rep = {"index": i, "halftop_dz": dz, "edge_ext": ext, "edge_lip": lip, "halftop_xf": XF, "halftop_ye": YE,
           "world": str(world), "world_tree_sha256": wh, "states": str(states),
           "static_iou": float(np.mean(static)), "robot_iou": float(np.mean([r["mean_iou"] for r in responses])),
           "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("EDGE_DONE", i, dz, ext, lip, round(rep["static_iou"], 4), round(rep["robot_iou"], 4),
          round(rep["loss"], 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.index is not None:
        return worker(a.index)
    OUT.mkdir(exist_ok=False)
    v12 = json.loads(V12_FREEZE.read_text())
    files = dict(v12["files"])
    for p, hsh in files.items():
        if sha(p) != hsh:
            raise ValueError(f"v12 evidence changed {p}")
    (OUT / "freeze.json").write_text(json.dumps({
        "queries": v12["queries"], "grid_dz_ext_lip": GRID, "prev": str(PREV), "halftop_xf_ye": [XF, YE],
        "files": files, "script_sha256": sha(__file__), "v23_script_sha256": sha(v23.__file__), "GT_inputs": False,
        "derived_from": str(V12_FREEZE),
        "family_design_informed_by": "v28 Target outcome (only x+050_y-060 fails) + diag_rack_ym060_v28 (y-060 "
                                     "slide-off the -y outer edge); selection uses public RGB only",
        "data_roles": "fit evidence: v12 9 probes, dense v16 rear, dense v23/v25/v27/v30 (v22/v24/v26/v28 "
                      "positions); acceptance on NEW unused -y edge positions",
        "objective": "static IoU + v12 9-probe response here (front regression); dense score separately"}, indent=1))

    def run(i):
        with (OUT / f"cand{i:02d}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((i, r.returncode))
        return json.loads((OUT / f"cand{i:02d}" / "result.json").read_text())

    with ThreadPoolExecutor(18) as pool:
        rows = list(pool.map(run, range(len(GRID))))
    for p, hsh in files.items():
        if sha(p) != hsh:
            raise ValueError(f"input changed {p}")
    ref = json.loads((PREV / "result.json").read_text())
    zero = rows[GRID.index((.03, 0., 0.))]
    ident = {"static_iou": [zero["static_iou"], ref["static_iou"]], "robot_iou": [zero["robot_iou"], ref["robot_iou"]]}
    ok = all(abs(x - y) < 1e-6 for x, y in ident.values())
    brief = [{k: v for k, v in r.items() if k != "responses"} for r in rows]
    (OUT / "selection_v12_objective.json").write_text(json.dumps({"identity_check": ident, "identity_ok": ok,
                                                                  "candidates": brief}, indent=1))
    print("IDENTITY", "OK" if ok else "MISMATCH", ident, flush=True)
    print("EDGE_V30_DONE", flush=True)


if __name__ == "__main__":
    main()
