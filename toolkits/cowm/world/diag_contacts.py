"""S4 diagnostic: deepest contacts after settling, with body names (run inside the Builder sandbox)."""
import os

import numpy as np

from cowm.world.env import CodeWorldEnv

OUT = os.environ.get("COWM_WORLDS", "/data_cf/cowmpo/worlds")


def main():
    env = CodeWorldEnv(os.path.join(OUT, "s4_ref_goal_v0"), use_camera_obs=False, has_renderer=False,
                       has_offscreen_renderer=False)
    m, d = env.sim.model, env.sim.data
    hold = np.zeros(7)
    hold[6] = -1.0
    env.reset()
    for label, n in (("after reset", 0), ("after 5 s", 100)):
        for _ in range(n):
            env.step(hold)
        rows = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.body_id2name(m.geom_bodyid[c.geom1])
            b2 = m.body_id2name(m.geom_bodyid[c.geom2])
            rows.append((c.dist * 1000, b1, b2))
        rows.sort()
        print(f"== {label}: ncon={d.ncon}")
        for r in rows[:6]:
            print(f"   {r[0]:7.2f} mm  {r[1]}  <->  {r[2]}")


if __name__ == "__main__":
    main()
