"""S2 Code WM side: same action sequences on a bare robosuite Panda (no LIBERO import).

Uses only declared shared priors: robosuite Panda model, OSC_POSE controller, control_freq=20,
and the reset pose (arm + gripper qpos at t=0). Scene objects are absent: S2 tests the
action interface in free space, not contact.
"""
import argparse

import numpy as np

from actions import make_sequences


def build_env():
    import robosuite as suite
    from robosuite.environments.manipulation.lift import Lift

    return Lift(
        robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
        control_freq=20,
        initialization_noise=None,
        hard_reset=False,
    )


# Shared prior (plan §3.7A): fixed robot reset pose of the target platform.
# OSC nullspace torques pull joints toward this reference, NOT toward the episode init state:
# env.reset() sets controller.initial_joint to the reset pose, and loading an init state
# afterwards only changes qpos. Using the episode init state here removes a ~4 mm drift.
RESET_QPOS = np.array([0.0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])


def set_robot_state(env, s0, nullspace_ref):
    """Start state = what the policy observes after warmup. Executor hidden state set explicitly:
    OSC cache (forced refresh, no stale-goal transient) and the gripper command integrator."""
    robot = env.robots[0]
    env.sim.data.qvel[:] = 0.0
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = s0["arm_q"]
    env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = s0["grip_q"]
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = s0["arm_qvel"]
    env.sim.data.qvel[robot._ref_gripper_joint_vel_indexes] = s0["grip_qvel"]
    env.sim.forward()
    robot.gripper.current_action = np.array(s0["grip_current_action"], dtype=np.float64)
    robot.controller.update(force=True)
    robot.controller.update_initial_joints(nullspace_ref)
    robot.controller.reset_goal()


def robot_state(env):
    obs = env._get_observations(force_update=True)
    robot = env.robots[0]
    return {
        "eef_pos": obs["robot0_eef_pos"].copy(),
        "eef_quat": obs["robot0_eef_quat"].copy(),
        # robosuite Lift only exposes joint cos/sin; read joint angles from qpos directly
        "joint_pos": env.sim.data.qpos[robot._ref_joint_pos_indexes].copy(),
        "gripper_qpos": obs["robot0_gripper_qpos"].copy(),
        "base_pos": env.sim.data.get_body_xpos(robot.robot_model.root_body).copy(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset-from", required=True, help="target npz; only t=0 robot qpos is read")
    p.add_argument("--out", required=True)
    p.add_argument("--nullspace-ref", choices=["reset", "init"], default="reset",
                   help="OSC nullspace reference: fixed reset pose (target semantics) or episode init state")
    args = p.parse_args()

    # S2 tests the executor only: the post-warmup robot state is read from the target as a test input.
    # In the full Code WM this comes from the M1 state estimator (step S5), not from the target.
    ref = np.load(args.reset_from)
    s0 = {
        "arm_q": ref["zero/joint_pos"][0],
        "grip_q": ref["zero/gripper_qpos"][0],
        "arm_qvel": ref["zero/arm_qvel0"],
        "grip_qvel": ref["zero/grip_qvel0"],
        "grip_current_action": ref["zero/grip_current_action"],
    }

    env = build_env()
    env.reset()
    record = {}
    for name, actions in make_sequences().items():
        env.reset()
        set_robot_state(env, s0, RESET_QPOS if args.nullspace_ref == "reset" else s0["arm_q"])
        states = [robot_state(env)]
        for a in actions:
            env.step(a)
            states.append(robot_state(env))
        for key in states[0]:
            record[f"{name}/{key}"] = np.stack([s[key] for s in states])
    np.savez(args.out, **record)
    print("saved", args.out)


if __name__ == "__main__":
    main()
