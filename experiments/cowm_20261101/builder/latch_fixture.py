# Copyright 2026 The RLinf Authors.
"""Opt-in latching (hysteresis) joint detent, computed from physical qpos at every step.

Codex's bistable_fixture switches wells at the midpoint in both directions. v51/v52 showed this cannot fit the public
stove probes: weak pushes (model q .3-.48) must stay off, while a turn to ~.65 that the gripper then drags back must
stay on. A latch does both: lock to the upper well once q > s_up, unlock only when q < s_down (s_down < s_up).
World declaration: objects[*]["joint_latch"] = {"stable_positions": [lo, hi], "s_up": .., "s_down": .., "stiffness": k,
  optional "k_off" (spring while unlatched), "k_on" (while latched), "fl_back_mult" (one-way: frictionloss x mult
  while turning back towards off)}
"""
from cowm.world import env as runtime
from articulation_contacts import ArticulatedEnv


class LatchEnv(ArticulatedEnv):
    def _load_model(self):
        super()._load_model()
        self._latch_cache, self._latched = None, {}

    def _reset_internal(self):
        super()._reset_internal()
        self._latched = {}

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        m, d = self.sim.model, self.sim.data
        if self._latch_cache is None:
            entries = []
            for obj, decl in zip(self.objects, self.world["objects"]):
                spec = decl.get("joint_latch")
                if spec is None:
                    continue
                root = m.body_name2id(obj.root_body)
                joints = [j for j in range(m.njnt) if int(m.body_rootid[m.jnt_bodyid[j]]) == root]
                if len(joints) != 1 or not m.jnt_limited[joints[0]]:
                    raise ValueError("latch requires one limited fixture joint")
                j = joints[0]
                lo, hi = map(float, spec["stable_positions"])
                up, down = float(spec["s_up"]), float(spec["s_down"])
                k = (float(spec.get("k_off", spec["stiffness"])), float(spec.get("k_on", spec["stiffness"])))
                if not (m.jnt_range[j, 0] <= lo < down < up < hi <= m.jnt_range[j, 1] and min(k) > 0):
                    raise ValueError("invalid latch")
                dof = int(m.jnt_dofadr[j])
                back = float(spec.get("fl_back_mult", 1.))
                entries.append((j, int(m.jnt_qposadr[j]), lo, hi, up, down, k, dof, float(m.dof_frictionloss[dof]), back))
            self._latch_cache = entries
        for j, addr, lo, hi, up, down, k, dof, fl0, back in self._latch_cache:
            # one-way (ratchet-like) structure: extra Coulomb friction when turning back towards off
            m.dof_frictionloss[dof] = fl0 * (back if float(d.qvel[dof]) < 0 else 1.)
            q = float(d.qpos[addr])
            on = self._latched.get(j, False)
            if on and q < down:
                on = False
            elif not on and q > up:
                on = True
            self._latched[j] = on
            m.jnt_stiffness[j] = k[1] if on else k[0]
            m.qpos_spring[addr] = hi if on else lo


def install():
    runtime.CodeWorldEnv = LatchEnv
