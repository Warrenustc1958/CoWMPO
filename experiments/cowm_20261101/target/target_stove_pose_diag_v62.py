"""Target-private diagnostic: Builder stove initial state (policy states) vs s5_gt stove AABB, t7_i0..i5. Not used for fitting."""
import json
from pathlib import Path
import numpy as np
from cowm.init.aabb import object_aabb
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
B = Path("/data_cf/cowmpo/worlds/codex_stove_hold_g2_20261031")
W = B / "stove_rotating_dial_v1/world0"
G = Path("/data_cf/cowmpo/target_eval/s5_gt")
out = {}
sc = Scene(str(W)); k = sc.names.index("stove")
for i in range(6):
    ep = f"libero_goal_t7_i{i}"
    sd, rd = ("stove_policy_fit_states_v1", "stove_policy_fit_robot_zero_v1") if i >= 3 else ("stove_policy_states_v1", "stove_policy_robot_zero_v1")
    sc.reset()
    apply_state(sc, json.loads((B / sd / ep / "state.json").read_text())["objects"], require_matched=True,
                robot=json.loads((B / rd / ep / "robot.json").read_text())["robot"])
    c, h = object_aabb(sc.env.sim.model, sc.env.sim.data, np.flatnonzero(sc.geom2obj == k))
    gt = json.loads((G / ("dev3" if i < 3 else "fit3") / f"{ep}.json").read_text())["stove"]
    d = (c - sc.base) - np.asarray(gt["aabb_center_rel_base"])
    out[ep] = {"dxyz_cm": (d * 100).round(2).tolist(), "xy_cm": round(float(np.linalg.norm(d[:2]) * 100), 2)}
    print(ep, out[ep], flush=True)
sc.env.close()
T = Path("/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/stove_pose_diag_v62"); T.mkdir(parents=True, exist_ok=True)
(T / "summary.json").write_text(json.dumps(out, indent=1))
