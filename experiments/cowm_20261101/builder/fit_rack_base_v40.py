# Copyright 2026 The RLinf Authors.
"""Rack near-end base trim v40: v36 cand03 (v31 cand18 + far-end slats) with the low base plate's near end (local -x,
robot side) trimmed by t; top surfaces and every contact part unchanged.
Why: v39 dense front-row probes (vert_x-080_*, valid gripper-only contacts) show the near top surface exists where the
model has it (model 2-4 steps early, within tolerance), so the top cannot be trimmed; the remaining S5 x offset can
only come from unprobed low geometry. Selection (public only, fixed in freeze.json): max mean public static IoU over
60 dev32+fit28; guardrail: transition onsets (dec0.5) on all 36 public probes unchanged vs v36c03 (max |delta| <= 1).
usage: python fit_rack_base_v40.py [--index i]
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

import fit_rack_far_v36 as v36
import fit_rack_panel_v23 as v23
from cowm.init.aabb import object_aabb
from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
BASE = R / "rack_scale_scan_dev_v8/scale00"
PREV = R / "rack_far_dev_v36/cand03"
OUT = R / "rack_base_dev_v40"
GRID = [0., .02, .035, .05]
DIRS = v36.DIRS
RGBA = v36.RGBA


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out_dir, width, depth, height, incline, t):
    orig = v23._wrap

    def wrap(name, assets, body, top, radius):
        old = v23._part('box', [depth / 2, width / 2, .004], [0, 0, .004], RGBA, density=600., friction=.7)
        assert body.count(old) == 1, "base plate not found"
        if t > 0:
            new = v23._part('box', [(depth - t) / 2, width / 2, .004], [t / 2, 0, .004], RGBA, density=600., friction=.7)
            body = body.replace(old, new)
        return orig(name, assets, body, top, radius)
    v23._wrap = wrap
    try:
        return v36.build(out_dir, width, depth, height, incline, .07, 3, .008)
    finally:
        v23._wrap = orig


def worker(i):
    t = GRID[i]
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(PREV / "world", world)
    meta = json.loads((world / "world.json").read_text())
    next(o for o in meta["objects"] if o["name"] == "rack")["attrs"].update(base_trim_near=float(t))
    build(world / "assets/rack", W, D, H, base["incline"], t)
    meta.pop("world_sha256", None)
    meta["rack_base_v40"] = {"prev": str(PREV), "trim": t, "rule": "v36 cand03 + near-end base plate trimmed"}
    meta["world_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    (world / "world.json").write_text(json.dumps(meta, indent=1))
    wh = tree_sha(world)
    scene = Scene(str(world))
    k = scene.names.index("rack")
    rows = []
    for role, (sd, rd, md) in DIRS.items():
        for ep in sorted(p.parent.name for p in (R / sd).glob("*/state.json")):
            scene.reset()
            apply_state(scene, json.loads((R / sd / ep / "state.json").read_text())["objects"], require_matched=True,
                        robot=json.loads((R / rd / ep / "robot.json").read_text())["robot"])
            c, h = object_aabb(scene.env.sim.model, scene.env.sim.data, np.flatnonzero(scene.geom2obj == k))
            with np.load(R / md / ep / "masks.npz") as a:
                s = iou(scene.seg() == k, a["rack"])
            rows.append({"episode": ep, "role": role, "public_iou": float(s),
                         "aabb_center_rel_base": (c - scene.base).tolist(), "aabb_half": h.tolist()})
    scene.env.close()
    verify_tree(world, wh)
    rep = {"index": i, "base_trim_near": t, "world": str(world), "world_tree_sha256": wh,
           "states": json.loads((PREV / "result.json").read_text())["states"],
           "public_iou_all": float(np.mean([r["public_iou"] for r in rows])), "n": len(rows), "rows": rows}
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("BASE_DONE", i, t, round(rep["public_iou_all"], 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.index is not None:
        return worker(a.index)
    OUT.mkdir(exist_ok=False)
    (OUT / "freeze.json").write_text(json.dumps({
        "grid_trim": GRID, "prev": str(PREV), "script_sha256": sha(__file__), "GT_inputs": False,
        "selection": "max mean public static IoU over 60 dev32+fit28",
        "guardrail": "dec0.5 onsets on all 36 public probes (27 old + far34 + front39) within 1 step of v36c03",
        "family_design_informed_by": "S5 Target diagnostic direction + v39 public front-row contacts",
        "data_roles": "fit evidence only"}, indent=1))

    def run(i):
        with (OUT / f"cand{i:02d}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((i, r.returncode))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(run, range(len(GRID))))
    print("BASE_V40_DONE", flush=True)


if __name__ == "__main__":
    main()
