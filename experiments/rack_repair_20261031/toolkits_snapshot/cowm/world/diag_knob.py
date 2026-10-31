"""Diagnostic: stove knob response to applied torque, plus contacts touching the knob body."""
import os

import numpy as np

from cowm.world.env import CodeWorldEnv

W = os.path.join(os.environ.get("COWM_WORLDS", "/data_cf/cowmpo/worlds"), "s4_ref_goal_v02")
HOLD = np.array([0, 0, 0, 0, 0, 0, -1.0])


def main():
    env = CodeWorldEnv(W, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    m, d = env.sim.model, env.sim.data
    jid = m.joint_name2id("stove_knob")
    dof, qa = m.jnt_dofadr[jid], m.jnt_qposadr[jid]
    bid = m.jnt_bodyid[jid]
    print("knob body", m.body_id2name(bid), "mass", m.body_mass[bid], "inertia", m.body_inertia[bid],
          "damping", m.dof_damping[dof], "armature", m.dof_armature[dof], "frictionloss", m.dof_frictionloss[dof],
          "range", m.jnt_range[jid], "limited", m.jnt_limited[jid] if hasattr(m, "jnt_limited") else "?")
    for tau in (0.0, 0.005, 0.02, 0.1, 0.3):
        env.reset()
        d.qfrc_applied[dof] = tau
        qs = []
        for _ in range(40):
            env.step(HOLD)
            qs.append(float(d.qpos[qa]))
        d.qfrc_applied[dof] = 0.0
        touching = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if bid in (b1, b2):
                touching.add((m.body_id2name(b1), m.body_id2name(b2), round(c.dist * 1000, 2)))
        print(f"tau={tau}: q after 10/20/40 steps = {qs[9]:.4f} {qs[19]:.4f} {qs[39]:.4f}  contacts={touching}")


if __name__ == "__main__":
    main()
