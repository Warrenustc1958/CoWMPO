"""Reset order-independence test (Step 3, CPU/OSMesa). Exit code 1 on any failure.

  fixture_baseline : after apply_state(B) (fixtures moved through model.body_pos) a plain reset() puts every
                     fixture back to the compiled pose
  a_b_a            : episode A -> episode B -> episode A in ONE scene: the second A equals the first A bitwise
                     (agentview RGB + qpos/qvel per step, including the 15-step warmup start)
  b_then_a_fresh   : a fresh scene that runs B first, then A, gives the same A as a scene that runs A first
usage: python -m cowm.world.test_reset_order <world> <state_root> <episode_A> <episode_B>
"""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state

ACT = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, -1.0])


def episode(scene, state, n=20):
    scene.reset()
    apply_state(scene, state)
    out = [scene.rgb().astype(np.int16).ravel()]
    for k in range(n):
        obs, _, _, _ = scene.env.step(ACT if k < n // 2 else -ACT)
        d = scene.env.sim.data
        out.append(np.concatenate([obs["agentview_image"].astype(np.int16).ravel(), d.qpos, d.qvel]))
    return out


def same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def main():
    world, state_root, ea, eb = sys.argv[1:5]
    load = lambda e: json.load(open(os.path.join(state_root, e, "state.json")))["objects"]
    sa, sb = load(ea), load(eb)
    res = {}
    s1 = Scene(world)
    m = s1.env.sim.model
    s1.reset()
    base = (m.body_pos.copy(), m.body_quat.copy())
    apply_state(s1, sb)
    moved = not np.array_equal(m.body_pos, base[0])
    s1.reset()
    res["fixture_baseline"] = moved and np.array_equal(m.body_pos, base[0]) and np.array_equal(m.body_quat, base[1])
    a1 = episode(s1, sa)
    episode(s1, sb)
    a2 = episode(s1, sa)
    res["a_b_a"] = same(a1, a2)
    s1.env.close()
    s2 = Scene(world)
    episode(s2, sb)
    a3 = episode(s2, sa)
    res["b_then_a_fresh"] = same(a1, a3)
    s2.env.close()
    ok = all(res.values())
    print(" ".join(f"{k}={v}" for k, v in res.items()), "fixtures_moved_by_B", moved)
    print("RESET_ORDER_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
