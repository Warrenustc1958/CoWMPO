# Copyright 2026 The RLinf Authors.
"""Freeze the rear-repaired rack model and plan a NEW independent acceptance probe set (v22). Builder side, no GT.

Frozen model: rack_post_dev_v21/cand07 (selected by the dense rear score v16: -y rear post at .45w raised 2 cm,
top bar from the -y post to the centre line, +y post moved to .25w; front slats, sides, origin unchanged).
Probes: rows x+040 and x+120 (mm from the Builder rack AABB centre) x columns y-080/+000/+080. These positions
were never used for planning, fitting or any Target run. Actions are generated exactly like
plan_rack_vertical_dev_v3 (same planning world/state/robot, closed loop on the Builder eef only), so the probe
design does not depend on the model under test.
Acceptance rule (fixed here, before any Target execution):
  valid  : Target has gripper-rack contact in the contact phase, no arm contact, no other-object contact, no
           pre-contact (judged on Target contact types only; invalid probes are reported, never dropped)
  pass   : |first contact step model - Target| <= 5 AND |eef (robot base frame) at first contact| <= 2 cm
  set    : ACCEPT iff >= 4 probes valid and every valid probe passes; the unrepaired v8 model is run on the same
           probes for reference only (not used for any decision)
usage: python plan_rack_accept_v22.py [--key k]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cowm import roles
from cowm.init.aabb import object_aabb
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
OUT = R / "rack_accept_plan_v22"
EP = "libero_goal_t0_i0"
BASE = R / "bowl_support_dev_v1/world11"                       # same planning world as v3
STATE = R / "natural_states_all_dev_v1" / EP / "state.json"
ROBOT = R / "robot_rgb_all_dev_v2" / EP / "robot.json"
FROZEN = R / "rack_post_dev_v21/cand07"
REFERENCE = R / "rack_scale_scan_dev_v8/scale00"
SCORE = R / "rack_dense_score_v16/score.json"
SPECS = {f"acc_x{dx:+04d}_y{dy:+04d}": (dx, dy) for dx in (40, 120) for dy in (-80, 0, 80)}
USED = ["vert_x-080_y-080", "vert_x-080_y+000", "vert_x-080_y+080", "vert_x+000_y-080", "vert_x+000_y+000",
        "vert_x+000_y+080", "vert_x+080_y-080", "vert_x+080_y+000", "vert_x+080_y+080"]
RULE = {"tol_steps": 5, "tol_cm": 2.0, "min_valid": 4, "set": "ACCEPT iff >= min_valid valid and all valid pass"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan(key):
    roles.require([EP], {"dev"}, "rack acceptance probe planning")
    dest = OUT / key
    dest.mkdir()
    scene = Scene(str(BASE))
    scene.reset()
    apply_state(scene, json.loads(STATE.read_text())["objects"], require_matched=True,
                robot=json.loads(ROBOT.read_text())["robot"])
    m = scene.env.sim.model
    top = scene.world["table"]["height"]
    center, _ = object_aabb(m, scene.env.sim.data, np.nonzero(scene.geom2obj == scene.names.index("rack"))[0])
    dx, dy = SPECS[key]
    start = center[:2] + np.array([dx, dy]) / 1000
    acts, phases = [], {}

    def drive(target, n, phase):
        for _ in range(n):
            a = np.zeros(7)
            a[:3] = np.clip(8 * (np.asarray(target) - scene.env.eef_pos()), -.65, .65)
            a[6] = 1
            scene.env.step(a)
            acts.append(list(map(float, a)))
        phases[phase] = len(acts)
    drive([*scene.env.eef_pos()[:2], top + .60], 80, "raise")
    drive([*start, top + .60], 100, "approach")
    drive([*start, top + .08], 160, "contact")
    drive([*scene.env.eef_pos()[:2], top + .60], 80, "retract")
    scene.env.close()
    (dest / "probe.json").write_text(json.dumps({
        "episode": EP, "obj": "rack", "mode": "isolated_contact", "key": key, "actions": acts,
        "frames": list(range(0, len(acts) + 1, 10)), "phases": phases, "offset_mm": [dx, dy],
        "world_tree_sha256": tree_sha(BASE), "state_sha256": sha(STATE), "robot_record_sha256": sha(ROBOT),
        "purpose": "independent acceptance (never used for planning/fitting/Target before); generated like v3"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    a = ap.parse_args()
    if a.key:
        return plan(a.key)
    assert not set(SPECS) & set(USED)
    sel = json.loads(SCORE.read_text())["candidates"][0]
    if sel["cand"] != "rack_post_dev_v21__cand07":
        raise ValueError(f"dense score top changed: {sel['cand']}")
    res = json.loads((FROZEN / "result.json").read_text())
    verify_tree(res["world"], res["world_tree_sha256"])
    OUT.mkdir(exist_ok=False)
    inputs = {str(p): sha(p) for p in (STATE, ROBOT, SCORE, FROZEN / "result.json", REFERENCE / "result.json")}
    (OUT / "freeze.json").write_text(json.dumps({
        "frozen_model": {"result": str(FROZEN / "result.json"), "world": res["world"],
                         "world_tree_sha256": res["world_tree_sha256"], "states": res["states"],
                         "dense_score": {k: sel[k] for k in ("params", "static_iou", "roi_iou_mean", "dcy_abs_max",
                                                             "stall_err_max", "loss")}},
        "reference_model": str(REFERENCE), "planning_world": {str(BASE): tree_sha(BASE)}, "grid_mm": SPECS,
        "previously_used_positions": USED, "rule": RULE, "inputs": inputs, "script_sha256": sha(__file__),
        "GT_inputs": False, "Target_queries_executed": 0,
        "data_roles": "fit evidence so far: 9 v3 positions (4 original + 5 v11) incl. dense rear v16; these 6 "
                      "positions are acceptance-only until their Target result is seen, then they become fit evidence"},
        indent=1))

    def run(key):
        with (OUT / f"{key}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--key", key], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((key, r.returncode))
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(run, SPECS))
    for p, h in inputs.items():
        if sha(p) != h:
            raise ValueError(f"input changed {p}")
    print("ACCEPT_PLAN_V22_DONE", sorted(SPECS), flush=True)


if __name__ == "__main__":
    main()
