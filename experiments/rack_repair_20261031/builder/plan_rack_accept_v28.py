# Copyright 2026 The RLinf Authors.
"""Freeze the selected v29 rack model and plan a NEW independent acceptance probe set (v28). Builder side, no GT.

Frozen model: SELECTED = top of the dense score v29 over 21 public probes (rear16 + acc23 + acc25 + acc27), chosen
with the static IoU drop and the front-row x-080 regression (v12 objective) checked; see NOTES. The +y edge residual
is recorded separately, not a repair target.
Probes: x+030 / x+130 x y-020 / y+020, plus x+050_y-060 / x+100_y-020 (mm from the Builder rack AABB centre).
None of these 6 positions was used for planning, fitting, scoring or any Target run (USED_MM lists all 27 used ones).
Actions are generated exactly like v22 / plan_rack_vertical_dev_v3 (same planning world/state/robot, closed loop
on the Builder eef only), so the probe design does not depend on the model under test.
Acceptance rule (identical to v22, fixed here before any Target execution):
  valid  : Target has gripper-rack contact in the contact phase, no arm contact, no other-object contact, no
           pre-contact (judged on Target contact types only; invalid probes are reported, never dropped)
  pass   : |first contact step model - Target| <= 5 AND |eef (robot base frame) at first contact| <= 2 cm
  set    : ACCEPT iff >= 4 probes valid and every valid probe passes; v25c01, v23c03, v21c07 and the unrepaired v8
           model are run on the same probes for reference only (not used for any decision)
usage: python plan_rack_accept_v28.py [--key k]
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
SELECTED = "rack_halftop_dev_v29__cand10"   # dense score v29 top (loss .7117, ex-y060 .6738)
NOTES = {
    "family": "v23 cand03 + flat -y half-top plate dz .03, xf 0, ye -.10 W; no rotation/shift (v29)",
    "loss": "v29c10 .7117 vs v25c01 .7285 / v23c03 .7537 / v21c07 .8831 / v12 .9803 (21 public probes); "
            "same top without the two y-060 slide-off probes (.6738)",
    "neg_y_iou": "mean ROI IoU on -y probes .851 vs v25c01 .819 / v23c03 .805; acc27 .852 vs .806",
    "static_iou_drop": "v29c10 .6810 vs v23c03 .7035 (-.0225) / v25c01 .6987",
    "front_row_x-080": "v12-objective responses v29c10 vs v23c03: y-080 .8555 vs .8493, y+000/y+080 identical; "
                       "no regression",
    "rear16": ".921 vs .927 (v23c03/v25c01)",
    "y+edge_residual": "recorded separately, not a repair target",
    "y-060_probes": "diag_rack_ym060_v28: real slide-off the -y outer edge; ROI IoU kept, stall/dcy not used",
    "tracker_lag": "v7b masks lag ROI entry by ~3-8 steps on every probe (common to all models)"}
OUT = R / "rack_accept_plan_v28"
EP = "libero_goal_t0_i0"
BASE = R / "bowl_support_dev_v1/world11"                       # same planning world as v3 / v22
STATE = R / "natural_states_all_dev_v1" / EP / "state.json"
ROBOT = R / "robot_rgb_all_dev_v2" / EP / "robot.json"
FROZEN = R / SELECTED.replace("__", "/")
REFERENCES = {"v25c01": R / "rack_side_dev_v25/cand01", "v23c03": R / "rack_panel_dev_v23/cand03",
              "v21c07": R / "rack_post_dev_v21/cand07", "v8": R / "rack_scale_scan_dev_v8/scale00"}
SCORE = R / "rack_dense_score_v29/score.json"
SPECS = {f"acc28_x{dx:+04d}_y{dy:+04d}": (dx, dy) for dx, dy in
         [(30, -20), (30, 20), (130, -20), (130, 20), (50, -60), (100, -20)]}
USED_MM = [(dx, dy) for dx in (-80, 0, 80) for dy in (-80, 0, 80)] + \
    [(dx, dy) for dx in (40, 120) for dy in (-80, 0, 80)] + [(dx, dy) for dx in (60, 80, 100) for dy in (-40, 40)] + \
    [(20, -40), (20, 40), (110, -40), (110, 40), (70, -60), (90, -60)]
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
        "purpose": "independent acceptance v28 (never used for planning/fitting/Target before); generated like v22-v26"},
        indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    a = ap.parse_args()
    if a.key:
        return plan(a.key)
    assert not set(SPECS.values()) & set(USED_MM)
    sel = json.loads(SCORE.read_text())["candidates"][0]
    if sel["cand"] != SELECTED:
        raise ValueError(f"dense score top changed: {sel['cand']}")
    res = json.loads((FROZEN / "result.json").read_text())
    verify_tree(res["world"], res["world_tree_sha256"])
    OUT.mkdir(exist_ok=False)
    inputs = {str(p): sha(p) for p in (STATE, ROBOT, SCORE, FROZEN / "result.json",
                                       *(q / "result.json" for q in REFERENCES.values()))}
    (OUT / "freeze.json").write_text(json.dumps({
        "frozen_model": {"result": str(FROZEN / "result.json"), "world": res["world"],
                         "world_tree_sha256": res["world_tree_sha256"], "states": res["states"],
                         "dense_score": {k: sel[k] for k in ("params", "static_iou", "roi_iou_mean", "roi_iou_rear16",
                                                             "roi_iou_acc23", "roi_iou_acc25", "roi_iou_acc27",
                                                             "roi_iou_neg_y_all", "stall_err_acc27_neg_y",
                                                             "dcy_abs_max", "stall_err_max", "loss", "loss_ex_ym060")}},
        "reference_models": {k: str(p) for k, p in REFERENCES.items()},
        "planning_world": {str(BASE): tree_sha(BASE)}, "grid_mm": SPECS,
        "previously_used_positions_mm": USED_MM, "rule": RULE, "inputs": inputs, "script_sha256": sha(__file__),
        "selection_notes": NOTES,
        "GT_inputs": False, "Target_queries_executed": 0,
        "data_roles": "fit evidence so far: 9 v3 positions + dense rear v16 + v22 (dense v23) + v24 (dense v25) + v26 "
                      "(dense v27) positions; Target first contact seen for v22/v24/v26; the v27/v27b/v29 families were "
                      "informed by the v24/v26 -y outcome. These 6 v28 positions are acceptance-only "
                      "until their Target result is seen; afterwards they become fit evidence"},
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
    print("ACCEPT_PLAN_V28_DONE", sorted(SPECS), flush=True)


if __name__ == "__main__":
    main()
