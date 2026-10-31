# Copyright 2026 The RLinf Authors.
"""Builder diagnostic (no GT, no Target numbers): which rack geom the gripper touches first in the frozen v23 cand03
model on the 6 v24 probe actions, and where (rack-local x/y, contact z above the table).
usage: python diag_rack_first_contact_v25.py
"""
import json
from pathlib import Path

import numpy as np

from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
PLAN = R / "rack_accept_plan_v24"
CAND = R / "rack_panel_dev_v23/cand03"
OUT = R / "diag_rack_first_contact_v25.json"
EP = "libero_goal_t0_i0"


def main():
    res = json.loads((CAND / "result.json").read_text())
    sc = Scene(res["world"])
    m = sc.env.sim.model
    k = sc.names.index("rack")
    grip = np.array([(m.body_id2name(int(m.geom_bodyid[g])) or "").startswith("gripper0") for g in range(m.ngeom)])
    doc = json.loads((Path(res["states"]) / EP / "state.json").read_text())
    robot = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    top = sc.world["table"]["height"]
    rows = []
    for key in sorted(json.loads((PLAN / "freeze.json").read_text())["grid_mm"]):
        probe = json.loads((PLAN / key / "probe.json").read_text())
        sc.reset()
        apply_state(sc, doc["objects"], require_matched=True, robot=robot)
        d = sc.env.sim.data
        rb = int(m.geom_bodyid[np.nonzero(sc.geom2obj == k)[0][0]])
        R0, p0 = d.body_xmat[rb].reshape(3, 3).copy(), d.body_xpos[rb].copy()
        rec = {"key": key, "first_step": None}
        for step, act in enumerate(probe["actions"], 1):
            sc.env.step(np.asarray(act))
            hit = [(int(c.geom1), int(c.geom2), c.pos.copy()) for c in d.contact[:d.ncon]
                   if (grip[int(c.geom1)] and sc.geom2obj[int(c.geom2)] == k) or
                   (grip[int(c.geom2)] and sc.geom2obj[int(c.geom1)] == k)]
            if hit:
                g1, g2, pos = hit[0]
                rg = g2 if sc.geom2obj[g2] == k else g1
                loc = R0.T @ (pos - p0)
                rec.update(first_step=step, rack_geom=rg, geom_size=m.geom_size[rg].round(4).tolist(),
                           geom_local_pos=(R0.T @ (d.geom_xpos[rg] - p0)).round(4).tolist(),
                           contact_local=loc.round(4).tolist(), contact_z_above_table=round(float(pos[2] - top), 4),
                           eef_z_above_table=round(float(sc.env.eef_pos()[2] - top), 4))
                break
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    sc.env.close()
    OUT.write_text(json.dumps({"model": str(CAND), "GT_inputs": False, "Target_numbers_read": False, "rows": rows},
                              indent=1))


if __name__ == "__main__":
    main()
