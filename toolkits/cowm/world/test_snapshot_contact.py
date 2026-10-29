"""Snapshot/restore across contact phases with the policy's RGB input (Step 3, CPU/OSMesa). Exit 1 on failure.

Uses a real strike probe (Builder-planned actions) on an S5 state. Phases are derived from the probe's own
trajectory (per-step gripper-object contact flag and object speed), then each is ASSERTED at the snapshot:
  pre     : before the first contact            -> no contact
  contact : first contact step + 1              -> contact active
  slide   : object speed > SLIDE_V after contact -> object moving
  post    : first step after the contact window with no contact -> no contact (reported as not observed if the
            probe never releases the object; then the test fails rather than relabelling an in-contact step)
For each phase: run 15 actions -> SA (agentview RGB, qpos, qvel, done per step); restore; run a different
random branch; restore; rerun -> must equal SA bitwise.
Not covered here (no grasp in a strike probe): grasping / holding / release phases.
usage: python test_snapshot_contact.py <world> <state_root> <probe_root> <episode>
"""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state

SLIDE_V = 0.02   # m/s
HORIZON = 15


def contact(scene, k_obj):
    d = scene.env.sim.data
    for c in d.contact[:d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if (scene.geom2obj[g1] == k_obj and scene.robot_geom[g2]) or (scene.geom2obj[g2] == k_obj and scene.robot_geom[g1]):
            return True
    return False


def run(scene, acts):
    out = []
    for a in acts:
        obs, _, done, _ = scene.env.step(np.asarray(a))
        d = scene.env.sim.data
        out.append(np.concatenate([obs["agentview_image"].astype(np.float64).ravel(), d.qpos, d.qvel, [float(done)]]))
    return out


def main():
    world, state_root, probe_root, ep = sys.argv[1:5]
    state = json.load(open(os.path.join(state_root, ep, "state.json")))["objects"]
    probe = json.load(open(os.path.join(probe_root, ep, "probe.json")))
    acts, obj = probe["actions"], probe["obj"]
    scene = Scene(world)
    k = scene.names.index(obj)
    dt = scene.env.control_timestep
    # 1) trajectory of the probe: contact flag and object speed per step (step i = after action i)
    scene.reset()
    apply_state(scene, state)
    p_prev = scene.env.object_pos(obj).copy()
    flags, speed = [False], [0.0]
    for a in acts:
        scene.env.step(np.asarray(a))
        p = scene.env.object_pos(obj).copy()
        flags.append(contact(scene, k))
        speed.append(float(np.linalg.norm(p[:2] - p_prev[:2]) / dt))
        p_prev = p
    last_ok = len(acts) - HORIZON
    kc = next((i for i, f in enumerate(flags) if f), None)
    res = {"first_contact_step": kc}
    if kc is None:
        print(json.dumps(res), "no gripper-object contact in this probe")
        print("SNAPSHOT_CONTACT_TEST FAIL")
        sys.exit(1)
    ke = max(i for i, f in enumerate(flags) if f)          # last contact step
    pts = {"pre": max(kc - 3, 1), "contact": kc + 1 if flags[kc + 1] else kc,
           "slide": next((i for i in range(kc, last_ok) if speed[i] > SLIDE_V), None),
           "post": next((i for i in range(ke + 1, last_ok) if not flags[i]), None)}
    expect = {"pre": lambda s: not flags[s], "contact": lambda s: flags[s], "slide": lambda s: speed[s] > SLIDE_V,
              "post": lambda s: not flags[s]}
    res["contact_window"] = [kc, ke]
    rng = np.random.default_rng(0)
    ok = True
    for name, s in pts.items():
        if s is None or s > last_ok:
            res[name] = {"observed": False}
            ok = False
            continue
        scene.reset()
        apply_state(scene, state)
        run(scene, acts[:s])
        live = {"in_contact": contact(scene, k), "speed": round(speed[s], 4)}
        phase_ok = bool(expect[name](s)) and (live["in_contact"] == flags[s])
        snap = scene.env.snapshot()
        sa = run(scene, acts[s:s + HORIZON])
        scene.env.restore(snap)
        run(scene, [np.concatenate([rng.uniform(-1, 1, 6) * 0.5, [1.0]]) for _ in range(10)])   # other branch
        scene.env.restore(snap)
        sb = run(scene, acts[s:s + HORIZON])
        eq = all(np.array_equal(x, y) for x, y in zip(sa, sb))
        res[name] = {"step": s, **live, "phase_ok": phase_ok, "equal": eq}
        ok &= eq and phase_ok
    scene.env.close()
    res["not_covered"] = ["grasp", "hold", "release", "KIR (backend supports_kir=False)"]
    print(json.dumps(res))
    print("SNAPSHOT_CONTACT_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
