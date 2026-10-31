# Copyright 2026 The RLinf Authors.
"""Rack back-only repair v12: keep the frozen v8 model (front, sides, origin, poses) and ADD rear structure only.

Evidence (all public, Builder side): six dev first-frame rack silhouettes + contact-phase robot masks of 9
gripper-only probes (4 original isolated/vertical with v7b tracks; 5 vertical exported in
codex_rack_vertical_dev_v11 with v7b tracks). The 5 new probes are FIT evidence from now on (their Target
first-contact numbers were seen in the v11 held-out check), not acceptance.
Candidates: rear extension e (m) behind the original back face and rear raise h (m) above the original height;
rear parts = 2 posts at the new back edge, a top rail at height+h, and a base rail when e > 0. Everything of the
original model is unchanged, so the robot-facing front and the sides cannot move.
Score: 1 - static IoU + 3 x (1 - mean contact-phase robot IoU over the 9 probes). No Target data.
usage: python fit_rack_back_v12.py [--index i]
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
OUT = R / "rack_back_dev_v12"
QCFG = R / "rack_clean_pose_dev_v6/freeze.json"
V7B = R / "rack_robot_redetect_v7b"
V11 = R / "rack_vertical_v11_tracks/frames"
PLAN = R / "rack_vertical_preflight_dev_v3"
NEW_KEYS = ["vert_x-080_y-080", "vert_x-080_y+000", "vert_x+080_y-080", "vert_x+080_y+000", "vert_x+080_y+080"]
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
GRID = list(itertools.product([0.0, 0.03, 0.05, 0.07], [0.0, 0.03, 0.06]))   # (e, h)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contact_frames(probe):
    ph = list(probe["phases"])
    i = ph.index("contact")
    start, end = probe["phases"][ph[i - 1]], probe["phases"]["contact"]
    return [s for s in probe["frames"] if start < s <= end]


def queries():
    qs = [dict(q, video=str(V7B / q["key"])) for q in json.loads(QCFG.read_text())["queries"]]
    for k in NEW_KEYS:
        src = PLAN / k / "probe.json"
        qs.append({"group": "vertical_v11", "key": k, "source": str(src), "video": str(V11 / k),
                   "frames": contact_frames(json.loads(src.read_text()))})
    return qs


def build_back(out_dir, width, depth, height, incline, e, h, density=600., friction=.7):
    """Original slatted two-shelf body (experimental_rack.build, unchanged) + rear parts behind x = depth/2."""
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
    for side in [-1, 1]:
        body += part('box', [.008, .008, height / 2], [.25 * depth, side * .35 * width, height / 2])
    # ---- rear-only additions (v12) ----
    xb = depth / 2 + e - .008                   # new back edge (rod centre)
    hb = height + h
    if e > 0 or h > 0:
        for side in [-1, 1]:
            body += part('box', [.008, .008, hb / 2], [xb, side * .35 * width, hb / 2])
        body += part('box', [.008, width / 2, .008], [xb, 0, hb - .008])
    if e > 0:
        body += part('box', [e / 2, width / 2, .004], [depth / 2 + e / 2, 0, .004])
    return _save(out_dir, 'rack', _wrap('rack', '', body, hb, max(width, depth + e) / 2))


def worker(i):
    e, h = GRID[i]
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    inc = base["incline"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(base["world"], world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(back_ext=float(e), back_raise=float(h))
    ob["top_z"] = float(H + h)
    build_back(world / "assets/rack", W, D, H, inc, e, h)
    meta.pop("world_sha256")
    meta["rack_back_v12"] = {"e": e, "h": h, "base": str(BASE), "rule": "rear-only additions; front/sides/origin unchanged"}
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
    rep = {"index": i, "e": e, "h": h, "world": str(world), "world_tree_sha256": wh, "states": str(states),
           "static_iou": float(np.mean(static)), "robot_iou": float(np.mean([r["mean_iou"] for r in responses])),
           "robot_iou_new5": float(np.mean([r["mean_iou"] for r in responses if r["group"] == "vertical_v11"])),
           "robot_iou_old4": float(np.mean([r["mean_iou"] for r in responses if r["group"] != "vertical_v11"])),
           "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("BACK_DONE", i, e, h, round(rep["static_iou"], 4), round(rep["robot_iou_old4"], 4),
          round(rep["robot_iou_new5"], 4), round(rep["loss"], 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.index is not None:
        return worker(a.index)
    OUT.mkdir(exist_ok=False)
    qs = queries()
    files = {str(BASE / "result.json"): sha(BASE / "result.json")}
    for q in qs:
        files[q["source"]] = sha(q["source"])
        for s in q["frames"]:
            for n in ("masks.npz", "association.json"):
                p = Path(q["video"]) / f"{EP}__f{s:03d}" / n
                files[str(p)] = sha(p)
    (OUT / "freeze.json").write_text(json.dumps({
        "queries": qs, "grid_e_h_m": GRID, "files": files, "script_sha256": sha(__file__), "GT_inputs": False,
        "data_roles": "the 5 v11 probes are fit evidence (Target first-contact seen in v11 check); a NEW held-out "
                      "probe set is required for acceptance",
        "objective": "1 - static IoU + 3 x (1 - contact-phase robot IoU), 9 probes, v7b tracks"}, indent=1))

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
    best = min(brief, key=lambda r: r["loss"])
    (OUT / "selection.json").write_text(json.dumps({"best": best, "candidates": brief}, indent=1))
    print("BACK_SELECTED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
