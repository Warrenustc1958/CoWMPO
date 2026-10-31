# Copyright 2026 The RLinf Authors.
"""Near-end horizontal rack probes v38 (FIT evidence, Builder side, no GT). Vertical near-end probes (v37) were invalid:
the forearm hits the rack before the gripper (arm-rack contact 27-90 steps). Here the gripper leads: raise, move to
STANDOFF before the Builder rack AABB centre on the robot side (-x), descend to table + h (below the rack top), then
push along +x to centre - STOP. Same closed-loop eef driver, planning world, state and robot as plan_rack_accept_v32;
frames dense over the push. Target exports public RGB only; validity from Target contact types only.
usage: python plan_rack_near_horiz_v38.py [--key k]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cowm import roles
from cowm.init.aabb import object_aabb
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
OUT = R / "rack_near_horiz_plan_v38"
EP = "libero_goal_t0_i0"
BASE = R / "bowl_support_dev_v1/world11"
STATE = R / "natural_states_all_dev_v1" / EP / "state.json"
ROBOT = R / "robot_rgb_all_dev_v2" / EP / "robot.json"
STANDOFF, STOP = .26, .02
SPECS = {f"near38_h{int(h * 1000):03d}_y{dy:+04d}": (h, dy) for h in (.10, .16) for dy in (-40, 0, 40)}
PHASES = (("raise", 80), ("approach", 100), ("descend", 80), ("contact", 200), ("retract", 80))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan(key):
    roles.require([EP], {"dev"}, "rack near-end horizontal probe planning")
    dest = OUT / key
    dest.mkdir()
    scene = Scene(str(BASE))
    scene.reset()
    apply_state(scene, json.loads(STATE.read_text())["objects"], require_matched=True,
                robot=json.loads(ROBOT.read_text())["robot"])
    m = scene.env.sim.model
    top = scene.world["table"]["height"]
    center, _ = object_aabb(m, scene.env.sim.data, np.nonzero(scene.geom2obj == scene.names.index("rack"))[0])
    h, dy = SPECS[key]
    y = center[1] + dy / 1000
    acts, phases = [], {}

    def drive(target, n, phase):
        for _ in range(n):
            a = np.zeros(7)
            a[:3] = np.clip(8 * (np.asarray(target) - scene.env.eef_pos()), -.65, .65)
            a[6] = 1
            scene.env.step(a)
            acts.append(list(map(float, a)))
        phases[phase] = len(acts)
    n = dict(PHASES)
    drive([*scene.env.eef_pos()[:2], top + .45], n["raise"], "raise")
    drive([center[0] - STANDOFF, y, top + .45], n["approach"], "approach")
    drive([center[0] - STANDOFF, y, top + h], n["descend"], "descend")
    drive([center[0] - STOP, y, top + h], n["contact"], "contact")
    drive([center[0] - STANDOFF, y, top + h], n["retract"], "retract")
    scene.env.close()
    a0 = phases["descend"]
    frames = sorted(set(range(0, len(acts) + 1, 10)) | set(range(a0, a0 + 121)))
    (dest / "probe.json").write_text(json.dumps({
        "episode": EP, "obj": "rack", "mode": "isolated_contact", "key": key, "actions": acts, "frames": frames,
        "phases": phases, "height_m": h, "offset_y_mm": dy, "dense_window": [a0, a0 + 120],
        "world_tree_sha256": tree_sha(BASE), "state_sha256": sha(STATE), "robot_record_sha256": sha(ROBOT),
        "purpose": "fit evidence: near-end horizontal active probe (S5 extent); gripper leads"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    a = ap.parse_args()
    if a.key:
        return plan(a.key)
    OUT.mkdir(exist_ok=False)
    for k in SPECS:
        r = subprocess.run([sys.executable, __file__, "--key", k], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError((k, r.stderr[-800:]))
        p = json.loads((OUT / k / "probe.json").read_text())
        print("NEARH_PLAN", k, len(p["frames"]), p["dense_window"], flush=True)
    (OUT / "freeze.json").write_text(json.dumps({"keys": sorted(SPECS), "specs": SPECS, "standoff": STANDOFF,
                                                 "stop": STOP, "GT_inputs": False,
                                                 "data_roles": "fit evidence (S5 near-end extent)"}, indent=1))
    print("NEARH_PLAN_V38_DONE", flush=True)


if __name__ == "__main__":
    main()
