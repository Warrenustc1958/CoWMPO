# Copyright 2026 The RLinf Authors.
"""Freeze the selected v31 rack model and plan a NEW independent acceptance probe set (v32). Builder side, no GT.

Frozen model: SELECTED = top of the transition score v36 over 33 public probes (rear16 + acc23 + acc25 + acc27), chosen
with the static IoU drop and the front-row x-080 regression (v12 objective) checked; see NOTES. The +y edge residual
is recorded separately, not a repair target.
Probes: x+040 / x+060 / x+110 y-060, x+050 / x+090 y-040, control x+070_y+040 (mm from the Builder rack AABB centre).
None of these 6 positions was used for planning, fitting, scoring or any Target run (USED_MM lists all 33 used ones).
Actions are generated exactly like v22 / plan_rack_vertical_dev_v3 (same planning world/state/robot, closed loop
on the Builder eef only), so the probe design does not depend on the model under test.
Acceptance rule (identical to v22, fixed here before any Target execution):
  valid  : Target has gripper-rack contact in the contact phase, no arm contact, no other-object contact, no
           pre-contact (judged on Target contact types only; invalid probes are reported, never dropped)
  pass   : |first contact step model - Target| <= 5 AND |eef (robot base frame) at first contact| <= 2 cm
  set    : ACCEPT iff >= 4 probes valid and every valid probe passes; v25c01, v23c03, v21c07 and the unrepaired v8
           model are run on the same probes for reference only (not used for any decision)
usage: python plan_rack_accept_v50.py [--key k]
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
SELECTED = "rack_hfield_v48__iter5"   # contact-surface fit, iteration 3
NOTES = {
    "method": "visual geometry of v36c03 unchanged; upper-tier collision replaced by invisible 1 cm collision "
              "columns (prior = max of 5 rays) + 3 counterexample-guided residual layers fitted on public RGB "
              "deceleration onsets of 42 public probes (WorldCoder / GIF-MCTS style repair loop), LOO-CV per layer",
    "public_transition": "52 valid probes (54 incl. v44+v47 dense): mean 1.885 -> 0.712 steps, max 6 -> 2, >=3 steps 14 -> 0 (in-sample); residuals at eef local xy; lam grid incl. .05/.1; 5 layers",
    "loo_last_layer": "0.717 cm mean, 1/40 over 2 cm (out-of-sample estimate)",
    "public_static_iou": ".66187 unchanged (columns transparent)",
    "excluded": "acc26_x+070_y-060, acc41_x+020_y-060 (no blocked moment; slide-off), rule fixed before fitting",
    "rejected_before": "v41: v36c03 3/6 (eef 2.3-2.8 cm)"}
OUT = R / "rack_accept_plan_v50"
EP = "libero_goal_t0_i0"
BASE = R / "bowl_support_dev_v1/world11"                       # same planning world as v3 / v22
STATE = R / "natural_states_all_dev_v1" / EP / "state.json"
ROBOT = R / "robot_rgb_all_dev_v2" / EP / "robot.json"
FROZEN = R / SELECTED.replace("__", "/")
REFERENCES = {"v46i4": R / "rack_hfield_v46/iter4", "v43i3": R / "rack_hfield_v43/iter3", "v36c03": R / "rack_far_dev_v36/cand03", "v31c18": R / "rack_rib_dev_v31/cand18", "v29c10": R / "rack_halftop_dev_v29/cand10", "v25c01": R / "rack_side_dev_v25/cand01", "v23c03": R / "rack_panel_dev_v23/cand03",
              "v21c07": R / "rack_post_dev_v21/cand07", "v8": R / "rack_scale_scan_dev_v8/scale00"}
SCORE = R / "rack_hfield_v48/score_v49.json"
SPECS = {f"acc50_x{dx:+04d}_y{dy:+04d}": (dx, dy) for dx, dy in
         [(80, -50), (100, -50), (-20, -40), (160, -60), (20, 70), (130, 60)]}
USED_MM = [(dx, dy) for dx in (-80, 0, 80) for dy in (-80, 0, 80)] + \
    [(dx, dy) for dx in (40, 120) for dy in (-80, 0, 80)] + [(dx, dy) for dx in (60, 80, 100) for dy in (-40, 40)] + \
    [(20, -40), (20, 40), (110, -40), (110, 40), (70, -60), (90, -60), (30, -20), (30, 20), (130, -20), (130, 20), (50, -60), (100, -20),
     (40, -60), (60, -60), (110, -60), (50, -40), (90, -40), (70, 40), (150, 0), (170, 0), (190, 0), (170, -50),
     (170, 50), (150, -50), (-100, 0), (-120, 0), (-140, 0), (-110, -50), (-110, 50), (-130, -50),
     (160, 20), (180, -30), (140, 40), (100, 60), (20, -60), (120, -40),
     (10, -30), (70, -20), (90, 50), (130, -50), (150, 30), (60, 60),
     (0, -50), (50, 20), (110, -20), (140, -10), (40, 40), (80, 70),
     (30, -40), (60, -20), (100, 30), (120, 20), (-40, 0), (170, -20)]
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
        "purpose": "independent acceptance v50 (hard regions; never used for planning/fitting/Target before); generated like v22-v26"},
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
                         "transition_score": {k: sel[k] for k in ("trans_err_mean", "trans_err_max", "n_ge3", "n", "public_static_iou", "loo_last_layer")}},
        "reference_models": {k: str(p) for k, p in REFERENCES.items()},
        "planning_world": {str(BASE): tree_sha(BASE)}, "grid_mm": SPECS,
        "previously_used_positions_mm": USED_MM, "rule": RULE, "inputs": inputs, "script_sha256": sha(__file__),
        "selection_notes": NOTES,
        "GT_inputs": False, "Target_queries_executed": 0,
        "data_roles": "fit evidence so far: 9 v3 positions + dense rear v16 + v22 (dense v23) + v24 (dense v25) + v26 "
                      "(dense v27) positions; Target first contact seen for v22/v24/v26; the v27-v31 families were "
                      "informed by the v24/v26 -y outcome. These 6 v50 positions are acceptance-only "
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
    print("ACCEPT_PLAN_V50_DONE", sorted(SPECS), flush=True)


if __name__ == "__main__":
    main()
