"""S4 gate (run INSIDE the Builder sandbox): compile, lint rejection, settle, zero-action stillness,
determinism, articulation present. Leak audit runs separately on the host (Evaluator side)."""
import json
import os
import sys
import tempfile

import numpy as np

from cowm.world.compile_world import compile_world
from cowm.world.lint import LintError, lint

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("COWM_WORLDS", "/data_cf/cowmpo/worlds")

BAD = {
    "import": "import os\ndef build(w):\n    pass\n",
    "open()": "def build(w):\n    open('/etc/passwd')\n",
    "private attr": "def build(w):\n    w._objects.append(1)\n",
    "dunder escape": "def build(w):\n    x = w.__class__\n",
    "timeline lambda": "def build(w):\n    f = lambda t: t\n",
    "two functions": "def build(w):\n    pass\ndef step(s):\n    pass\n",
    "while loop": "def build(w):\n    while True:\n        pass\n",
}


def main():
    ok = True
    res = {}
    for name, src in BAD.items():
        try:
            lint(src)
            good = False
        except (LintError, SyntaxError):
            good = True
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] lint rejects {name}")

    wdir = compile_world(os.path.join(HERE, "ref_goal_program.py"), os.path.join(OUT, "s4_ref_goal_v0"))
    world = json.load(open(os.path.join(wdir, "world.json")))
    print(f"[PASS] compiled {len(world['objects'])} objects, {len(world['params'])} params, "
          f"sha={world['world_sha256'][:12]}")

    from cowm.world.env import CodeWorldEnv

    env = CodeWorldEnv(wdir, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    names = [o["name"] for o in world["objects"] if o["free"]]
    jn = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
    art = [j for j in jn if "drawer_" in j or j.endswith("knob")]
    good = len(art) == 4
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] articulation joints present: {art}")

    hold = np.zeros(7)
    hold[6] = -1.0
    env.reset()
    p0 = {n: env.object_pos(n) for n in names}
    for _ in range(80):          # 4 s settle
        env.step(hold)
    p1 = {n: env.object_pos(n) for n in names}
    for _ in range(20):          # 1 s more: should be still
        env.step(hold)
    p2 = {n: env.object_pos(n) for n in names}
    drift = {n: float(np.linalg.norm(p2[n] - p1[n]) * 1000) for n in names}
    settle = {n: float(np.linalg.norm(p1[n] - p0[n]) * 1000) for n in names}
    pen = float(env.min_contact_dist() * 1000)
    good = max(drift.values()) < 1.0 and pen > -3.0 and max(settle.values()) < 10.0
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] settle: max settle {max(settle.values()):.2f} mm, "
          f"final-1s drift {max(drift.values()):.3f} mm, min contact dist {pen:.2f} mm")
    res.update(settle_mm=settle, drift_mm=drift, min_contact_mm=pen)

    rng = np.random.default_rng(0)
    acts = np.clip(rng.normal(0, 0.3, size=(40, 7)), -1, 1)
    traj = []
    for _ in range(2):
        env.reset()
        q = []
        for a in acts:
            env.step(a)
            q.append(env.sim.data.qpos.copy())
        traj.append(np.stack(q))
    d = float(np.abs(traj[0] - traj[1]).max())
    good = d == 0.0
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] determinism: max |dqpos| over 40 random steps = {d:.2e}")
    res["determinism_max_dqpos"] = d

    with open(os.path.join(wdir, "s4_selftest.json"), "w") as f:
        json.dump(dict(res, ok=bool(ok)), f, indent=2)
    print("S4_SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
