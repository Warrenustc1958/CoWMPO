"""S3 diagnostic: do fixture body poses change across reset() while qpos is identical? Does seeding fix it?"""
import os

import numpy as np


def run(env, init_state, seed=None, wait=15):
    if seed is not None:
        env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    warm = np.zeros(7)
    warm[6] = -1.0
    for _ in range(wait):
        obs, _, _, _ = env.step(warm)
    m = env.env.sim.model
    return (np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.int16),
            m.body_pos.copy(), m.body_quat.copy())


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    st = suite.get_task_init_states(0)[0]
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    m = env.env.sim.model

    a, pa, qa = run(env, st)
    b, pb, qb = run(env, st)
    moved = np.where(np.abs(pa - pb).max(axis=1) > 1e-9)[0]
    print("unseeded: bodies whose model.body_pos changed:",
          [(m.body_id2name(i), np.round(np.abs(pa[i] - pb[i]).max() * 1000, 2)) for i in moved][:10], "(mm)")
    print("unseeded: frame px>25:", round(100.0 * (np.abs(a - b).max(axis=2) > 25).mean(), 3), "%")

    c, pc, _ = run(env, st, seed=123)
    d, pd, _ = run(env, st, seed=123)
    print("seeded 123 vs 123: max|d body_pos|", np.abs(pc - pd).max(),
          " frame px>25:", round(100.0 * (np.abs(c - d).max(axis=2) > 25).mean(), 3), "%",
          " max|d px|:", int(np.abs(c - d).max()))


if __name__ == "__main__":
    main()
