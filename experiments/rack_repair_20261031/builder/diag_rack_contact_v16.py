# Copyright 2026 The RLinf Authors.
"""Builder-side diagnostic (v16): where is the gripper and what does it touch in a candidate world during the
rear probe's descent. Prints per step: eef site xyz (m, world), and robot-vs-object contact geom pairs.
No Target/GT input; the candidate world is the Builder's own model.
usage: python diag_rack_contact_v16.py <cand_result.json> <probe_key> <step0> <step1>
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
    cand, key, s0, s1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    res = json.loads(Path(cand).read_text())
    sc = Scene(res["world"])
    doc = json.loads((Path(res["states"]) / EP / "state.json").read_text())
    robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    probe = json.loads((R / "rack_vertical_dense_plan_v16" / key / "probe.json").read_text())
    sc.reset()
    apply_state(sc, doc["objects"], require_matched=True, robot=robot0)
    sim = sc.env.sim
    m = sim.model
    site = next(i for i in range(m.nsite) if "grip_site" in (m.site_id2name(i) or "") and "cylinder" not in m.site_id2name(i))
    rack_body = m.body_name2id(next(o for o in sc.env.objects if o.name == "rack").root_body)
    print("site", m.site_id2name(site), "rack_body_xpos", np.round(sim.data.body_xpos[rack_body], 4).tolist())
    for step, act in enumerate(probe["actions"], 1):
        sc.env.step(np.asarray(act))
        if step < s0:
            continue
        pairs = set()
        for c in sim.data.contact[:sim.data.ncon]:
            a, b = m.geom_id2name(c.geom1) or str(c.geom1), m.geom_id2name(c.geom2) or str(c.geom2)
            if ("gripper" in a or "robot" in a or "finger" in a or "gripper" in b or "robot" in b or "finger" in b) \
                    and "table" not in a + b:
                pairs.add(f"{a}~{b}")
        print(step, np.round(sim.data.site_xpos[site], 4).tolist(), np.round(act[:3], 3).tolist(), sorted(pairs)[:4])
        if step >= s1:
            break
    sc.env.close()


if __name__ == "__main__":
    main()
