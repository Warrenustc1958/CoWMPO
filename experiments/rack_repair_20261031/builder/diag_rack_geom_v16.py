"""Builder-side: list rack geoms (local pos/size) and the gripper site in rack-local frame at given steps. No GT."""
import json, sys
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020"); EP = "libero_goal_t0_i0"
res = json.loads(Path(sys.argv[1]).read_text()); key = sys.argv[2]; at = [int(x) for x in sys.argv[3:]]
sc = Scene(res["world"]); doc = json.loads((Path(res["states"]) / EP / "state.json").read_text())
robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
probe = json.loads((R / "rack_vertical_dense_plan_v16" / key / "probe.json").read_text())
sc.reset(); apply_state(sc, doc["objects"], require_matched=True, robot=robot0)
sim = sc.env.sim; m = sim.model; d = sim.data
rb = m.body_name2id(next(o for o in sc.env.objects if o.name == "rack").root_body)
Rm = d.body_xmat[rb].reshape(3, 3); p0 = d.body_xpos[rb]
print("rack yaw deg", round(float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))), 2))
robot_base = d.body_xpos[m.body_name2id(sc.env.robots[0].robot_model.root_body)]
print("robot base in rack frame", np.round(Rm.T @ (robot_base - p0), 3).tolist())
for g in range(m.ngeom):
    n = m.geom_id2name(g) or ""
    if n.startswith("rack_g") and m.geom_contype[g]:
        print(n, "type", m.geom_type[g], "local", np.round(Rm.T @ (d.geom_xpos[g] - p0), 4).tolist(), "size", np.round(m.geom_size[g], 4).tolist())
site = m.site_name2id("gripper0_grip_site")
f2 = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "") == "gripper0_finger2_collision"][0]
for step, act in enumerate(probe["actions"], 1):
    sc.env.step(np.asarray(act))
    if step in at:
        print("step", step, "site local", np.round(Rm.T @ (d.site_xpos[site] - p0), 4).tolist(), "finger2 local", np.round(Rm.T @ (d.geom_xpos[f2] - p0), 4).tolist())
    if step >= max(at): break
sc.env.close()
