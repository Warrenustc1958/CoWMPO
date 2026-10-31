# Copyright 2026 The RLinf Authors.
"""Rack far-end repair v36 (slats; v35 solid plate+wall failed the static-IoU guardrail): v31 cand18 (accepted contacts, v32 6/6) plus a full-height far-end extension beyond local
x = D/2: a collision top plate from x = D/2 to D/2 + ef at H + dzf (full width) and a thin far wall from the table to
the plate. Why: v34 far-end probes (x+150..x+190, y-050..+050) are all valid gripper-rack contacts in Target (contact
types only) and the public RGB deceleration onset there is 216-221, i.e. a top surface at ~rack top height, while the
model has nothing beyond x = D/2 (5/6 no contact). This also explains the constant S5 x offset. Low base extensions
(v33) cannot be reached by the 8 cm descent and changed nothing. Selection (public RGB only, fixed in freeze.json):
min transition error over all 33 public probes; guardrails: transition error on the 27 old probes <= v31c18 + .1,
mean public static IoU over 60 dev32+fit28 >= v31c18 - .02. Index 0 = v31 cand18 unchanged.
usage: python fit_rack_far_v36.py [--index i]
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
import fit_rack_rib_v31 as v31
from cowm.init.aabb import object_aabb
from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
BASE = R / "rack_scale_scan_dev_v8/scale00"
PREV = R / "rack_rib_dev_v31/cand18"
OUT = R / "rack_far_dev_v36"
C18 = (.035, .01, .05, -.035, .085)
GRID = [(0., 0, 0.)] + [(ef, n, sw) for ef in (.07, .08) for n in (2, 3, 4) for sw in (.008, .016)]
DIRS = {"dev": ("natural_states_all_dev_v1", "robot_rgb_all_dev_v2", "stove_parts_states_dev_v1"),
        "fit": ("natural_bowl_fit_v2", "robot_rgb_fit_v2", "stove_parts_states_fit_v1")}
RGBA = '0.45 0.3 0.2 1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out_dir, width, depth, height, incline, ef, n, sw):
    """v31 cand18 + n thin collision slats (width sw along x, full rack width) over the far end, top at H; no wall."""
    orig = v23._wrap

    def wrap(name, assets, body, top, radius):
        if ef > 0:
            x0, zt = depth / 2, height
            for j in range(n):
                xc = x0 + ef * (j + .5) / n
                body += v23._part('box', [sw / 2, width / 2, .004], [xc, 0, zt - .004], RGBA,
                                  density=600., friction=.7)
        return orig(name, assets, body, top, radius + ef)
    v23._wrap = wrap
    try:
        return v31.build(out_dir, width, depth, height, incline, *C18)
    finally:
        v23._wrap = orig


def worker(i):
    ef, n, sw = GRID[i]
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(PREV / "world", world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(far_ext=float(ef), far_slats=int(n), far_slat_w=float(sw))
    build(world / "assets/rack", W, D, H, base["incline"], ef, n, sw)
    meta.pop("world_sha256", None)
    meta["rack_far_v36"] = {"prev": str(PREV), "ef": ef, "n": n, "sw": sw,
                            "rule": "v31 cand18 + far-end thin slats at top height, no wall"}
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
    rep = {"index": i, "far_ext": ef, "far_dz": [n, sw], "world": str(world), "world_tree_sha256": wh,
           "states": json.loads((PREV / "result.json").read_text())["states"],
           "public_iou_all": float(np.mean([r["public_iou"] for r in rows])), "n": len(rows), "rows": rows}
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("FAR_DONE", i, ef, n, sw, round(rep["public_iou_all"], 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.index is not None:
        return worker(a.index)
    OUT.mkdir(exist_ok=False)
    (OUT / "freeze.json").write_text(json.dumps({
        "grid_ef_n_sw": GRID, "why_v36": "v35 solid plate+wall: transition fixed but public static IoU -.053 > guardrail -.02", "prev": str(PREV), "script_sha256": sha(__file__), "GT_inputs": False,
        "selection": "min transition error (dec0.5) over all 33 public probes (27 old + 6 far34)",
        "guardrails": {"old27_trans_err_rise": .1, "public_static_iou_drop": .02},
        "family_design_informed_by": "v34 far probes: public RGB onset + Target contact TYPES (validity only); "
                                     "S5 Target diagnostic (constant x offset)",
        "data_roles": "far34 probes and dev32+fit28 public masks are fit evidence"}, indent=1))

    def run(i):
        with (OUT / f"cand{i:02d}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((i, r.returncode))
    with ThreadPoolExecutor(13) as pool:
        list(pool.map(run, range(len(GRID))))
    print("FAR_V36_DONE", flush=True)


if __name__ == "__main__":
    main()
