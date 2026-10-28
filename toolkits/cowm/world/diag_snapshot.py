"""Which part of the step record differs after restore (same env)? Prints first differing step per field."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

from cowm.world.env import CodeWorldEnv


def record(env, acts):
    out = []
    for a in acts:
        obs, _, done, _ = env.step(a)
        d = env.sim.data
        r = {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "grip": np.array([env.robots[0].gripper.current_action[0]]),
             "done": np.array([float(done)]), "time": np.array([d.time]), "timestep": np.array([env.timestep])}
        for k, v in obs.items():
            if not k.endswith("_image") and np.ndim(v) <= 1:
                r["obs:" + k] = np.ravel(np.asarray(v, dtype=np.float64)).copy()
        out.append(r)
    return out


def main():
    world = sys.argv[1]
    rng = np.random.default_rng(0)
    acts = lambda n: [np.concatenate([rng.uniform(-1, 1, 6) * 0.5, [rng.choice([-1.0, 1.0])]]) for _ in range(n)]
    pre, A = acts(25), acts(40)
    env = CodeWorldEnv(world, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                       horizon=len(pre) + len(A), ignore_done=False)
    env.reset()
    record(env, pre)
    snap = env.snapshot()
    SA = record(env, A)
    env.restore(snap)
    SB = record(env, A)
    for key in SA[0]:
        diffs = [i for i in range(len(A)) if not np.array_equal(SA[i][key], SB[i][key])]
        if diffs:
            i = diffs[0]
            print(f"DIFF {key}: first step {i}, n={len(diffs)}, max={np.abs(SA[i][key] - SB[i][key]).max():.3g}")
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
