"""S4 diagnostic: plate/bowl tilt and penetration after settling, with and without MuJoCo multiccd."""
import os

import mujoco
import numpy as np

from cowm.world.env import CodeWorldEnv

OUT = os.environ.get("COWM_WORLDS", "/data_cf/cowmpo/worlds")


def tilt_deg(env, name):
    obj = next(o for o in env.objects if o.name == name)
    R = env.sim.data.body_xmat[env.sim.model.body_name2id(obj.root_body)].reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))


def run(multiccd):
    env = CodeWorldEnv(os.path.join(OUT, "s4_ref_goal_v0"), use_camera_obs=False, has_renderer=False,
                       has_offscreen_renderer=False)
    if multiccd:
        env.sim.model._model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    hold = np.zeros(7)
    hold[6] = -1.0
    env.reset()
    for _ in range(100):
        env.step(hold)
    m, d = env.sim.model, env.sim.data
    per = {}
    for i in range(d.ncon):
        c = d.contact[i]
        b = {m.body_id2name(m.geom_bodyid[c.geom1]), m.body_id2name(m.geom_bodyid[c.geom2])}
        for n in ("plate_main", "bowl_main"):
            if n in b:
                per.setdefault(n, []).append(c.dist * 1000)
    print(f"multiccd={multiccd}: plate tilt {tilt_deg(env, 'plate'):.2f} deg, bowl tilt {tilt_deg(env, 'bowl'):.2f} deg")
    for n, v in per.items():
        print(f"   {n}: {len(v)} contacts, min dist {min(v):.2f} mm")


if __name__ == "__main__":
    run(False)
    run(True)
