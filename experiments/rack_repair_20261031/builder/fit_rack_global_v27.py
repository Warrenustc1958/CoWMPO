# Copyright 2026 The RLinf Authors.
"""Rack global repair v27: v25 cand01 geometry (unchanged parts) under two global parameters:
  roll  rotation about the +y bottom edge (y = +W/2, z = 0), lifting the -y side by ~W*sin(roll); +y edge fixed
  dy    rigid y offset of the whole rack (m)
Why: v24 + v26 acceptance (12 positions) show one pattern: every -y position is hit 1-7 steps earlier in Target than
in the model, every +y position passes. v17-v25 local rear patches did not fix it (v26: v25c01 3/6 = v23c03), so the
error looks global (-y half too low, roll, or y pose), not a local rear part. The family design is informed by the
v24/v26 Target outcomes (recorded in freeze.json); selection uses public RGB only: static first-frame IoU + v12
9-probe regression here, dense score over all 21 public probes in score_rack_dense_v27.py. A new unused probe set is
required for acceptance.
roll = 0, dy = 0 must reproduce v25 cand01 exactly (checked in main).
usage: python fit_rack_global_v27.py [--index i]
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import fit_rack_side_v25 as v25
from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
BASE = R / "rack_scale_scan_dev_v8/scale00"
PREV = R / "rack_side_dev_v25/cand01"
OUT = R / "rack_global_dev_v27"
V12_FREEZE = R / "rack_back_dev_v12/freeze.json"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
HF, HB, YE = .02, .005, .05                       # v25 cand01 -y board
GRID = [(roll, dy) for roll in (0., 3., 6., 9.) for dy in (-.01, 0., .01)]   # roll in degrees, dy in m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out_dir, width, depth, height, incline, roll_deg, dy):
    """v25 build at cand01 values, wrapped in nested bodies: shift dy, then roll about the +y bottom edge."""
    a = math.radians(roll_deg)
    orig = v25._wrap

    def wrap(name, assets, body, top, radius):
        # rotation about x by -a: a point at y = -W (relative to the +y edge pivot) rises by W*sin(a)
        nested = (f'<body pos="0 {dy:.5f} 0"><body pos="0 {width / 2:.5f} 0" euler="{-a:.6f} 0 0">'
                  f'<body pos="0 {-width / 2:.5f} 0">{body}</body></body></body>')
        return orig(name, assets, nested, top + width * math.sin(a), radius + abs(dy))
    v25._wrap = wrap
    try:
        return v25.build(out_dir, width, depth, height, incline, HF, HB, YE)
    finally:
        v25._wrap = orig


def worker(i):
    roll, dy = GRID[i]
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    inc = base["incline"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(base["world"], world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(post_raise_neg=v25.H_N, post_raise_pos=v25.HP, post_bar=v25.BAR, post_y_neg=v25.PN,
                       bar_y_neg=v25.YN, bar_end_pos=v25.PE, post_y_pos=v25.PY, panel_dz=v25.DZ, panel_x0=v25.X0,
                       panel_y_end=v25.YE, side_hf=HF, side_hb=HB, side_ye=YE, side_xf=v25.XF,
                       global_roll_deg=float(roll), global_dy=float(dy))
    ob["top_z"] = float(H + max(v25.H_N, v25.DZ, HF) + W * math.sin(math.radians(roll)))
    build(world / "assets/rack", W, D, H, inc, roll, dy)
    meta.pop("world_sha256")
    meta["rack_global_v27"] = {"base": str(BASE), "prev": str(PREV), "roll_deg": roll, "dy": dy,
                               "rule": "v25 cand01 geometry; roll about the +y bottom edge, then rigid y shift"}
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
    rep = {"index": i, "global_roll_deg": roll, "global_dy": dy, "side_hf": HF, "side_hb": HB, "side_ye": YE,
           "world": str(world), "world_tree_sha256": wh, "states": str(states),
           "static_iou": float(np.mean(static)), "robot_iou": float(np.mean([r["mean_iou"] for r in responses])),
           "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("GLOBAL_DONE", i, roll, dy, round(rep["static_iou"], 4), round(rep["robot_iou"], 4), round(rep["loss"], 4),
          flush=True)


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
        "queries": v12["queries"], "grid_roll_deg_dy": GRID, "prev": str(PREV), "side": [HF, HB, YE],
        "files": files, "script_sha256": sha(__file__), "v25_script_sha256": sha(v25.__file__), "GT_inputs": False,
        "derived_from": str(V12_FREEZE),
        "family_design_informed_by": "v24 + v26 Target first-contact outcomes (all -y positions early, all +y pass); "
                                     "selection itself uses public RGB only",
        "data_roles": "fit evidence: v12 9 probes, dense v16 rear, dense v23 (v22 positions), dense v25 (v24 "
                      "positions), dense v27 (v26 positions); acceptance needs a NEW unused probe set",
        "objective": "static IoU + v12 9-probe response here (front regression); dense score separately"}, indent=1))

    def run(i):
        with (OUT / f"cand{i:02d}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((i, r.returncode))
        return json.loads((OUT / f"cand{i:02d}" / "result.json").read_text())

    with ThreadPoolExecutor(12) as pool:
        rows = list(pool.map(run, range(len(GRID))))
    for p, hsh in files.items():
        if sha(p) != hsh:
            raise ValueError(f"input changed {p}")
    # identity check: roll 0, dy 0 must reproduce v25 cand01
    ref = json.loads((PREV / "result.json").read_text())
    zero = rows[GRID.index((0., 0.))]
    ident = {"static_iou": [zero["static_iou"], ref["static_iou"]], "robot_iou": [zero["robot_iou"], ref["robot_iou"]]}
    ok = all(abs(x - y) < 1e-6 for x, y in ident.values())
    brief = [{k: v for k, v in r.items() if k != "responses"} for r in rows]
    (OUT / "selection_v12_objective.json").write_text(json.dumps({"identity_check": ident, "identity_ok": ok,
                                                                  "candidates": brief}, indent=1))
    print("IDENTITY", "OK" if ok else "MISMATCH", ident, flush=True)
    print("GLOBAL_V27_DONE", flush=True)


if __name__ == "__main__":
    main()
