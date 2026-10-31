"""CodeWorldEnv snapshot/restore test (review item): branching and cross-instance restore must be bitwise.

Compared per step: full MuJoCo state (qpos, qvel, act), gripper integrator, every proprio observation, done.
1. same env      : run prefix, snapshot, run suffix A -> SA; restore, run A again -> must equal SA.
2. branch        : restore, run a different suffix B, restore, run A -> still SA (branches don't leak).
3. new env       : a fresh CodeWorldEnv of the same world, restore the snapshot, run A -> SA.
4. terminated    : run past the horizon (done=True, step() refuses), restore the pre-terminal snapshot,
                   run A -> SA (done is part of the snapshot, not left over from the branch).
5. runtime params: change friction/mass after the snapshot (S6 apply_params style), restore -> SA.
6. version       : restoring into a different world version is refused.
Exit code 1 on any failure.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

from cowm.world.env import CodeWorldEnv

def obs_vec(obs):
    """Every non-image observation, in a fixed key order (robosuite versions differ in key names)."""
    keys = sorted(k for k, v in obs.items() if not k.endswith("_image") and np.ndim(v) <= 1)
    return np.concatenate([np.ravel(np.asarray(obs[k], dtype=np.float64)) for k in keys])


def run(env, acts):
    out = []
    for a in acts:
        obs, _, done, _ = env.step(a)
        d = env.sim.data
        out.append(np.concatenate([d.qpos, d.qvel, d.act, [env.robots[0].gripper.current_action[0]],
                                   obs_vec(obs), [float(done)]]).copy())
    return np.stack(out)


def make(world, horizon):
    return CodeWorldEnv(world, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                        horizon=horizon, ignore_done=False)


def main():
    world = sys.argv[1]
    other = sys.argv[2] if len(sys.argv) > 2 else None
    rng = np.random.default_rng(0)
    acts = lambda n: [np.concatenate([rng.uniform(-1, 1, 6) * 0.5, [rng.choice([-1.0, 1.0])]]) for _ in range(n)]
    pre, A, B = acts(25), acts(40), acts(40)
    H = len(pre) + len(A)          # horizon ends exactly after prefix + A
    res = {}
    env = make(world, H)
    env.reset()
    run(env, pre)
    snap = env.snapshot()
    SA = run(env, A)
    res["done_at_horizon"] = bool(env.done)
    env.restore(snap)
    res["same_env"] = np.array_equal(run(env, A), SA)
    env.restore(snap)
    run(env, B)
    env.restore(snap)
    res["branch"] = np.array_equal(run(env, A), SA)
    # terminated branch: run to the horizon, confirm step() is refused, then restore and continue
    env.restore(snap)
    run(env, B)
    try:
        env.step(B[0])
        res["refuses_after_done"] = False
    except ValueError:
        res["refuses_after_done"] = True
    env.restore(snap)
    try:
        res["terminated_restore"] = np.array_equal(run(env, A), SA)
    except ValueError:
        res["terminated_restore"] = False
    # runtime model edits after the snapshot must be undone by restore
    env.restore(snap)
    env.sim.model.geom_friction[:, 0] *= 0.5
    env.sim.model.body_mass[:] *= 1.3
    env.restore(snap)
    res["runtime_params"] = np.array_equal(run(env, A), SA)
    env2 = make(world, H)
    env2.reset()
    env2.restore(snap)
    S3 = run(env2, A)
    res["new_env"] = np.array_equal(S3, SA)
    res["max_diff_new_env"] = float(np.abs(S3 - SA).max())
    res["version_guard"] = True
    if other:
        env3 = make(other, H)
        try:
            env3.restore(snap)
            res["version_guard"] = False
        except ValueError:
            pass
    keys = ["done_at_horizon", "same_env", "branch", "refuses_after_done", "terminated_restore",
            "runtime_params", "new_env", "version_guard"]
    ok = all(res[k] for k in keys)
    print(" ".join(f"{k}={res[k]}" for k in keys), "max_diff_new_env", res["max_diff_new_env"])
    print("SNAPSHOT_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
