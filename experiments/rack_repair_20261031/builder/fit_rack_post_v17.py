# Copyright 2026 The RLinf Authors.
"""Rack rear-post repair v17: keep the frozen v8 model (front slats, sides, origin, poses) and change ONLY the
two rear posts (rack-local x = +0.25 depth, behind the centre, away from the robot at x < 0).

Why (Builder-side diagnostic diag_rack_contact_v16 / diag_rack_geom_v16, no GT): in probe vert_x+080_y-080 the
simulated finger lands on the 1.6 cm-wide top of post g38 and slips 4 cm sideways, while the dense public
tracks (codex_rack_vertical_dense_v16 + frozen v7b) show the real arm stopping ~2 steps earlier and not
slipping. v12/v15 rear additions were placed at the back edge (x >= 0.107) which the gripper never reaches.
Candidates: post raise h (m) and an optional top crossbar of half-depth b (m) joining the post tops.
Scores: static first-frame IoU (6 dev episodes) + v12 contact response (9 probes, regression of the front)
here; the dense rear score is computed by score_rack_dense_v16.py. No Target data.
usage: python fit_rack_post_v17.py [--index i]
"""
import argparse
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree
from cowm.world.templates import _part, _save, _wrap

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
BASE = R / "rack_scale_scan_dev_v8/scale00"
OUT = R / "rack_post_dev_v17"
V12_FREEZE = R / "rack_back_dev_v12/freeze.json"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
GRID = list(itertools.product([0.0, 0.01, 0.02, 0.03], [0.0, 0.008, 0.016]))   # (h, b)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_post(out_dir, width, depth, height, incline, h, b, density=600., friction=.7):
    """experimental_rack.build body with the two rear posts raised by h and, if b > 0, joined by a top bar."""
    rgba = '0.45 0.3 0.2 1'

    def part(kind, size, pos, euler=None):
        return _part(kind, size, pos, rgba, euler=euler, density=density, friction=friction)
    thickness = .008
    slope, co = math.sin(incline), math.cos(incline)
    upper = height - .45 * depth * abs(slope) - thickness
    lower = max(.05, .45 * upper)
    body = part('box', [depth / 2, width / 2, .004], [0, 0, .004])
    for level in [lower, upper]:
        for j in range(8):
            x = (-.45 + .9 * j / 7) * depth
            body += part('box', [.035 * depth, width / 2, thickness / 2], [co * x, 0, level - slope * x], euler=f'0 {incline} 0')
        body += part('box', [.012, width / 2, .006], [0, 0, level - .012])
    hp = height + h                              # ---- rear posts (the only changed parts) ----
    for side in [-1, 1]:
        body += part('box', [.008, .008, hp / 2], [.25 * depth, side * .35 * width, hp / 2])
    if b > 0:
        body += part('box', [b, .35 * width + .008, .008], [.25 * depth, 0, hp - .008])
    return _save(out_dir, 'rack', _wrap('rack', '', body, hp, max(width, depth) / 2))


def worker(i):
    h, b = GRID[i]
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    inc = base["incline"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(base["world"], world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(post_raise=float(h), post_bar=float(b))
    ob["top_z"] = float(H + h)
    build_post(world / "assets/rack", W, D, H, inc, h, b)
    meta.pop("world_sha256")
    meta["rack_post_v17"] = {"h": h, "b": b, "base": str(BASE), "rule": "rear posts only; slats/front/sides/origin unchanged"}
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
    rep = {"index": i, "h": h, "b": b, "world": str(world), "world_tree_sha256": wh, "states": str(states),
           "static_iou": float(np.mean(static)), "robot_iou": float(np.mean([r["mean_iou"] for r in responses])),
           "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("POST_DONE", i, h, b, round(rep["static_iou"], 4), round(rep["robot_iou"], 4), round(rep["loss"], 4), flush=True)


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
        "queries": v12["queries"], "grid_h_b_m": GRID, "files": files, "script_sha256": sha(__file__),
        "GT_inputs": False, "derived_from": str(V12_FREEZE),
        "data_roles": "same fit evidence as v12 + dense rear probes (fit); acceptance needs NEW unused probes "
                      "(x+040 / x+120 rows)",
        "objective": "static IoU + v12 9-probe response here (front regression); dense rear score separately"}, indent=1))

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
    brief = [{k: v for k, v in r.items() if k != "responses"} for r in rows]
    (OUT / "selection_v12_objective.json").write_text(json.dumps({"candidates": brief}, indent=1))
    print("POST_V17_DONE", flush=True)


if __name__ == "__main__":
    main()
