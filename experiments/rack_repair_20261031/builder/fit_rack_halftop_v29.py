# Copyright 2026 The RLinf Authors.
"""Rack -y half-top repair v29: v23 cand03 (rear posts + bar + rear top panel dz .005) plus ONE flat plate over the
-y half of the top at H + dz, from local x = xf to the back edge and from the -y edge to y = ye*width. No rigid
rotation, no shift: the footprint and the static silhouette stay (almost) unchanged.
Why: v24/v26 Target outcomes (all -y positions hit earlier, all +y pass) and the v27/v27b public dense scores (a
rigid roll fixes -y but shifts the top or the silhouette, so the top stays v25 cand01, already rejected at v26).
Builder check diag_rack_ym060_v28.json: the two acc26 y-060 probes slide off the -y outer edge (real, not tracking);
their ROI IoU stays in the fit, their stall/dcy are not used. Family design informed by those outcomes (recorded in
freeze.json); selection uses public RGB only: static IoU + v12 9-probe regression here, 21-probe dense score in
score_rack_dense_v29.py. Acceptance on the unused v28 positions.
usage: python fit_rack_halftop_v29.py [--index i]
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
PREV = R / "rack_panel_dev_v23/cand03"
OUT = R / "rack_halftop_dev_v29"
V12_FREEZE = R / "rack_back_dev_v12/freeze.json"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
DZ0, X00, YE0 = .005, 0., .35                     # v23 cand03 rear top panel
# -y half-top plate: (top above H (m), front edge local x (m), +y end as fraction of width)
GRID = [(dz, xf, ye) for dz in (.01, .02, .03, .04) for xf in (-.03, 0.) for ye in (-.10, 0.)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out_dir, width, depth, height, incline, dz, xf, ye):
    """v23 build_post at cand03 values plus the flat -y half-top plate (the only new part)."""
    orig = v23._wrap

    def wrap(name, assets, body, top, radius):
        xb, yl, yr, zt = depth / 2, -width / 2, ye * width, height + dz
        body += v23._part('box', [(xb - xf) / 2, (yr - yl) / 2, .004], [(xb + xf) / 2, (yr + yl) / 2, zt - .004],
                          '0.45 0.3 0.2 1', density=600., friction=.7)
        return orig(name, assets, body, max(top, zt), radius)
    v23._wrap = wrap
    try:
        return v23.build_post(out_dir, width, depth, height, incline, DZ0, X00, YE0)
    finally:
        v23._wrap = orig


def worker(i):
    dz, xf, ye = GRID[i]
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
                       panel_y_end=YE0, halftop_dz=float(dz), halftop_xf=float(xf), halftop_ye=float(ye))
    ob["top_z"] = float(H + max(v23.H_N, DZ0, dz))
    build(world / "assets/rack", W, D, H, inc, dz, xf, ye)
    meta.pop("world_sha256")
    meta["rack_halftop_v29"] = {"base": str(BASE), "prev": str(PREV), "dz": dz, "xf": xf, "ye": ye,
                                "rule": "v23 cand03 + flat -y half-top plate; no rotation/shift"}
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
    rep = {"index": i, "halftop_dz": dz, "halftop_xf": xf, "halftop_ye": ye, "panel_dz": DZ0,
           "world": str(world), "world_tree_sha256": wh, "states": str(states),
           "static_iou": float(np.mean(static)), "robot_iou": float(np.mean([r["mean_iou"] for r in responses])),
           "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("HALFTOP_DONE", i, dz, xf, ye, round(rep["static_iou"], 4), round(rep["robot_iou"], 4),
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
        "queries": v12["queries"], "grid_dz_xf_ye": GRID, "prev": str(PREV), "panel": [DZ0, X00, YE0],
        "files": files, "script_sha256": sha(__file__), "v23_script_sha256": sha(v23.__file__), "GT_inputs": False,
        "derived_from": str(V12_FREEZE),
        "family_design_informed_by": "v24 + v26 Target first-contact outcomes, v27/v27b public dense scores, "
                                     "diag_rack_ym060_v28 (y-060 slide-off); selection uses public RGB only",
        "data_roles": "fit evidence: v12 9 probes, dense v16 rear, dense v23/v25/v27 (v22/v24/v26 positions); "
                      "acceptance on the unused v28 positions",
        "objective": "static IoU + v12 9-probe response here (front regression); dense score separately"}, indent=1))

    def run(i):
        with (OUT / f"cand{i:02d}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((i, r.returncode))
        return json.loads((OUT / f"cand{i:02d}" / "result.json").read_text())

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(run, range(len(GRID))))
    for p, hsh in files.items():
        if sha(p) != hsh:
            raise ValueError(f"input changed {p}")
    brief = [{k: v for k, v in r.items() if k != "responses"} for r in rows]
    (OUT / "selection_v12_objective.json").write_text(json.dumps({"candidates": brief}, indent=1))
    print("HALFTOP_V29_DONE", flush=True)


if __name__ == "__main__":
    main()
