# Copyright 2026 The RLinf Authors.
"""Builder-side diagnostic (v22): for each probe of a plan dir, run a candidate world and report the first
gripper-rack contact step, the rack geom touched, and the gripper site in rack-local coordinates. No Target/GT.
usage: python diag_rack_first_contact_v22.py <cand_result.json> <plan_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
EP = "libero_goal_t0_i0"


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    plan = Path(sys.argv[2])
    robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    doc = json.loads((Path(res["states"]) / EP / "state.json").read_text())
    sc = Scene(res["world"])
    m = sc.env.sim.model
    k = sc.names.index("rack")
    grip = np.array([(m.body_id2name(int(m.geom_bodyid[g])) or "").startswith("gripper0") for g in range(m.ngeom)])
    site = m.site_name2id("gripper0_grip_site")
    for pj in sorted(plan.glob("*/probe.json")):
        probe = json.loads(pj.read_text())
        sc.reset()
        apply_state(sc, doc["objects"], require_matched=True, robot=robot0)
        d = sc.env.sim.data
        rb = m.body_name2id(next(o for o in sc.env.objects if o.name == "rack").root_body)
        Rm, p0 = d.body_xmat[rb].reshape(3, 3).copy(), d.body_xpos[rb].copy()
        hit = None
        for step, act in enumerate(probe["actions"], 1):
            sc.env.step(np.asarray(act))
            for c in d.contact[:d.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if grip[g1] and sc.geom2obj[g2] == k:
                    hit = (step, m.geom_id2name(g2), m.geom_id2name(g1))
                elif grip[g2] and sc.geom2obj[g1] == k:
                    hit = (step, m.geom_id2name(g1), m.geom_id2name(g2))
                if hit:
                    break
            if hit:
                break
        loc = np.round(Rm.T @ (d.site_xpos[site] - p0), 4).tolist()
        print(probe["key"], "first", hit, "site_local", loc, flush=True)
    sc.env.close()


if __name__ == "__main__":
    main()
