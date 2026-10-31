# Copyright 2026 The RLinf Authors.
"""Rack structure comparison v10: open-lattice hypothesis vs the slatted two-shelf model.

Step 1 (geometry): the frozen silhouette fitter (fit_rack_supported_silhouette.worker) with ONLY the structure
swapped (parts/build from experimental_rack_lattice): dims + incline + per-episode pose fitted to the six dev
rack silhouettes, 6 seeds, no Target data.
Step 2 (function): every seed is scored exactly like v8/v9 (static IoU + 3 x contact-phase robot IoU on the
four verified gripper-only probes, frozen v7b robot tracks). Writes rack_lattice_dev_v10/candidateNN.
usage: python fit_rack_lattice_v10.py [--seed i | --score i]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import experimental_rack_lattice as L
import fit_rack_supported_silhouette as F
from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
OUT = R / "rack_lattice_dev_v10"
TRACK = R / "rack_robot_redetect_v7b"
QCFG = R / "rack_clean_pose_dev_v6/freeze.json"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
SEEDS = 6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def geometry(seed):
    F.parts, F.build, F.OUT = L.parts, L.build, OUT      # structure swap only; fitter logic unchanged
    F.worker(seed)


def score(seed):
    dest = OUT / f"candidate{seed:02d}"
    fit = json.loads((dest / "fit.json").read_text())
    world = Path(fit["world"])
    wh = tree_sha(world)
    docs = {ep: json.loads((dest / "states" / ep / "state.json").read_text()) for ep in EPS}
    scene = Scene(str(world))
    static = []
    for ep in EPS:
        robot = json.loads((R / "robot_rgb_all_dev_v2" / ep / "robot.json").read_text())["robot"]
        scene.reset()
        apply_state(scene, docs[ep]["objects"], require_matched=True, robot=robot)
        k = scene.names.index("rack")
        with np.load(R / "visual_dev_bound_v1/dev3" / ep / "masks.npz") as a:
            target = a["rack"]
        pred = scene.seg() == k
        static.append(iou(pred, target))
        np.savez_compressed(dest / "states" / ep / "render_masks.npz", rack=pred)
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
            meta = json.loads((fr / "association.json").read_text())
            if meta["masks_sha256"] != sha(fr / "masks.npz") or meta["objects"]["robot"]["status"] != "matched":
                raise ValueError("invalid response evidence")
            with np.load(fr / "masks.npz") as a:
                vals.append(iou(scene.robot_mask(), a["robot"]))
        if len(vals) != len(q["frames"]):
            raise ValueError("incomplete contact response")
        responses.append({"key": q["key"], "mean_iou": float(np.mean(vals)), "ious": vals})
    scene.env.close()
    verify_tree(world, wh)
    rep = {"seed": seed, "structure": "open_lattice", "dims_m": fit["dims"], "incline": fit["incline"],
           "silhouette_mse": fit["loss"], "world": str(world), "world_tree_sha256": wh, "states": str(dest / "states"),
           "static_iou": float(np.mean(static)), "robot_iou": float(np.mean([r["mean_iou"] for r in responses])),
           "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("LATTICE_SCORED", seed, [round(v, 3) for v in fit["dims"]], round(rep["static_iou"], 4),
          round(rep["robot_iou"], 4), round(rep["loss"], 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--score", type=int)
    a = ap.parse_args()
    if a.seed is not None:
        return geometry(a.seed)
    if a.score is not None:
        return score(a.score)
    OUT.mkdir(exist_ok=False)
    qs = [dict(q, video=str(TRACK / q["key"])) for q in json.loads(QCFG.read_text())["queries"]]
    (OUT / "freeze.json").write_text(json.dumps({
        "queries": qs, "seeds": SEEDS, "structure": "open lattice (experimental_rack_lattice)",
        "lattice_sha256": sha(Path(L.__file__)), "fitter_sha256": sha(Path(F.__file__)), "script_sha256": sha(__file__),
        "objective": "1 - static IoU + 3 x (1 - contact-phase robot IoU); four verified gripper-only probes; v7b tracks",
        "GT_inputs": False}, indent=1))

    def run(i):
        for flag in ("--seed", "--score"):
            with (OUT / f"candidate{i:02d}{flag.replace('--', '_')}.log").open("w") as log:
                r = subprocess.run([sys.executable, __file__, flag, str(i)], stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                raise RuntimeError((i, flag, r.returncode))
        return json.loads((OUT / f"candidate{i:02d}" / "result.json").read_text())

    with ThreadPoolExecutor(SEEDS) as pool:
        rows = list(pool.map(run, range(SEEDS)))
    brief = [{k: v for k, v in r.items() if k != "responses"} for r in rows]
    best = min(brief, key=lambda r: r["loss"])
    (OUT / "selection.json").write_text(json.dumps({"best": best, "candidates": brief}, indent=1))
    print("LATTICE_SELECTED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
