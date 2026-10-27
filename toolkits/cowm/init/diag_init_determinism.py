"""S3 diagnostic: is reset -> set_init_state -> warmup deterministic across resets / processes?"""
import os
import sys

import numpy as np


def frame(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.int16)


def run(env, init_state, wait=15):
    env.reset()
    obs = env.set_init_state(init_state)
    warm = np.zeros(7)
    warm[6] = -1.0
    for _ in range(wait):
        obs, _, _, _ = env.step(warm)
    return frame(obs), env.env.sim.data.qpos.copy()


def cmp(name, a, b, qa=None, qb=None):
    d = np.abs(a - b)
    msg = (f"{name:28s} mean|d|={d.mean():7.3f}/255  max={d.max():3d}  "
           f"px>25: {100.0 * (d.max(axis=2) > 25).mean():6.3f}%")
    if qa is not None:
        msg += f"  max|dqpos|={np.abs(qa - qb).max():.2e}"
    print(msg)


def main():
    init_dir = sys.argv[1]
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    st = suite.get_task_init_states(0)[0]
    stored = np.load(os.path.join(init_dir, "libero_goal_t0_i0.npy"), allow_pickle=True)[0]["image"].astype(np.int16)

    e1 = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    a, qa = run(e1, st)
    b, qb = run(e1, st)
    e2 = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, ignore_done=True)
    c, qc = run(e2, st)
    cmp("stored vs run A (new env)", stored, a)
    cmp("run A vs run B (same env)", a, b, qa, qb)
    cmp("run A vs run C (2nd env)", a, c, qa, qc)
    # same env, explicit sim warmstart reset before loading the state
    e1.reset()
    e1.env.sim.data.qacc_warmstart[:] = 0
    d, qd = run(e1, st)
    cmp("run A vs run D (same env)", a, d, qa, qd)


if __name__ == "__main__":
    main()
