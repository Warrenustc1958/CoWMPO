"""S2 diagnostic: per-step EE motion under zero action in target LIBERO, and controller cache state."""
import os

import numpy as np


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = suite.get_task_init_states(0)
    env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_offscreen_renderer=False,
                     has_renderer=False, control_freq=20)

    for force_update in (False, True):
        env.reset()
        env.set_init_state(init_states[0])
        inner = env.env
        ctrl = inner.robots[0].controller
        print(f"\n== force_update={force_update}  new_update after set_init_state: {ctrl.new_update}")
        print("   controller cached ee_pos:", np.round(ctrl.ee_pos, 4))
        print("   sim ee_pos              :", np.round(inner._get_observations(force_update=True)["robot0_eef_pos"], 4))
        if force_update:
            ctrl.update(force=True)
            ctrl.reset_goal()
        prev = inner._get_observations(force_update=True)["robot0_eef_pos"].copy()
        steps = []
        for i in range(10):
            env.step(np.zeros(7))
            cur = inner._get_observations(force_update=True)["robot0_eef_pos"].copy()
            steps.append(np.linalg.norm(cur - prev) * 1000.0)
            prev = cur
        print("   per-step EE move (mm):", np.round(steps, 3))


if __name__ == "__main__":
    main()
