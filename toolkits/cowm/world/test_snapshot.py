"""CodeWorldEnv snapshot/restore test (review item): branching and cross-instance restore must be bitwise.

1. same env:   run prefix, snapshot, run suffix A -> states SA; restore, run A again -> must equal SA.
2. branch:     restore, run a different suffix B, restore, run A -> still equals SA (branches don't leak).
3. new env:    a fresh CodeWorldEnv of the same world, restore the snapshot, run A -> equals SA.
4. version:    restoring into a different world version is refused.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

from cowm.world.env import CodeWorldEnv


def run(env, acts):
    out = []
    for a in acts:
        env.step(a)
        d = env.sim.data
        out.append(np.concatenate([d.qpos, d.qvel, [env.robots[0].gripper.current_action[0]]]).copy())
    return np.stack(out)


def main():
    world = sys.argv[1]
    other = sys.argv[2] if len(sys.argv) > 2 else None
    rng = np.random.default_rng(0)
    acts = lambda n: [np.concatenate([rng.uniform(-1, 1, 6) * 0.5, [rng.choice([-1.0, 1.0])]]) for _ in range(n)]
    pre, A, B = acts(25), acts(40), acts(40)
    env = CodeWorldEnv(world, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    run(env, pre)
    snap = env.snapshot()
    SA = run(env, A)
    env.restore(snap)
    ok1 = np.array_equal(run(env, A), SA)
    env.restore(snap)
    run(env, B)
    env.restore(snap)
    ok2 = np.array_equal(run(env, A), SA)
    env2 = CodeWorldEnv(world, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    env2.reset()
    env2.restore(snap)
    S3 = run(env2, A)
    ok3 = np.array_equal(S3, SA)
    ok4 = True
    if other:
        env3 = CodeWorldEnv(other, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        try:
            env3.restore(snap)
            ok4 = False
        except ValueError:
            ok4 = True
    print("same_env", ok1, "branch", ok2, "new_env", ok3, "max_diff_new_env", float(np.abs(S3 - SA).max()),
          "version_guard", ok4)
    print("SNAPSHOT_TEST", "PASS" if ok1 and ok2 and ok3 and ok4 else "FAIL")


if __name__ == "__main__":
    main()
