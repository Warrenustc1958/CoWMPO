"""S2 target side: replay shared action sequences in real LIBERO-goal, record robot state.

Runs on the Target side only (it imports libero). CPU only: no camera, no renderer.
"""
import argparse
import os

import numpy as np

from actions import make_sequences


def robot_state(env):
    inner = env.env
    obs = inner._get_observations(force_update=True)
    robot = inner.robots[0]
    base = inner.sim.data.get_body_xpos(robot.robot_model.root_body).copy()
    return {
        "eef_pos": obs["robot0_eef_pos"].copy(),
        "eef_quat": obs["robot0_eef_quat"].copy(),
        "joint_pos": obs["robot0_joint_pos"].copy(),
        "gripper_qpos": obs["robot0_gripper_qpos"].copy(),
        "base_pos": base,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-id", type=int, default=0)
    p.add_argument("--num-steps-wait", type=int, default=15, help="RLinf LiberoEnv default warmup")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401  (registers problems)
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = suite.get_task_init_states(args.task_id)

    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
        control_freq=20,
    )
    env.reset()

    record = {"task": task.language, "init_id": args.init_id}
    for name, actions in make_sequences().items():
        # full reset per sequence: gripper current_action and controller state are not in sim state
        env.reset()
        env.set_init_state(init_states[args.init_id])
        # RLinf LiberoEnv.reset semantics: num_steps_wait zero-action steps, gripper -1 if reset_gripper_open.
        # The policy's first observation is the state AFTER this warmup.
        warm = np.zeros(7)
        warm[6] = -1.0
        for _ in range(args.num_steps_wait):
            env.step(warm)
        inner = env.env
        robot = inner.robots[0]
        record[f"{name}/init_qvel_norm"] = float(np.linalg.norm(inner.sim.data.qvel))
        record[f"{name}/arm_qvel0"] = inner.sim.data.qvel[robot._ref_joint_vel_indexes].copy()
        record[f"{name}/grip_qvel0"] = inner.sim.data.qvel[robot._ref_gripper_joint_vel_indexes].copy()
        record[f"{name}/grip_current_action"] = np.array(robot.gripper.current_action, dtype=np.float64)
        states = [robot_state(env)]
        for a in actions:
            env.step(a)
            states.append(robot_state(env))
        for key in states[0]:
            record[f"{name}/{key}"] = np.stack([s[key] for s in states])
        record[f"{name}/actions"] = actions
    np.savez(args.out, **record)
    print("saved", args.out, "task:", task.language)


if __name__ == "__main__":
    main()
