# Copyright 2026 The RLinf Authors.
"""Install an image-estimated Panda state with the shared controller reset semantics."""
import numpy as np

SIZES = {"arm_qpos": 7, "arm_qvel": 7, "grip_qpos": 2, "grip_qvel": 2,
         "grip_current_action": 2}


def robot_issues(record):
    """Return invalid fields; absent records retain the declared fixed-reset prior."""
    if record is None:
        return []
    if not isinstance(record, dict):
        return ["robot_state_not_mapping"]
    issues = []
    for field, size in SIZES.items():
        try:
            vector = np.asarray(record.get(field), dtype=float)
            valid = vector.shape == (size,) and np.isfinite(vector).all()
        except (ValueError, TypeError):
            valid = False
        if not valid:
            issues.append("invalid_" + field)
    return issues


def validate_robot_platform(scene, record):
    """Check a Panda record and destination platform before any state is installed."""
    issues = robot_issues(record)
    if issues:
        raise ValueError(f"invalid robot initialization: {issues}")
    if record is None:
        return
    robot = scene.env.robots[0]
    index = {"arm_qpos": robot._ref_joint_pos_indexes,
             "grip_qpos": robot._ref_gripper_joint_pos_indexes,
             "arm_qvel": robot._ref_joint_vel_indexes,
             "grip_qvel": robot._ref_gripper_joint_vel_indexes}
    for field, ids in index.items():
        if len(ids) != SIZES[field]:
            raise ValueError(f"robot platform dimension mismatch for {field}")
    if np.shape(robot.gripper.current_action) != (SIZES["grip_current_action"],):
        raise ValueError("robot platform gripper action dimension mismatch")


def install_robot_state(scene, record):
    """Install validated joint state, gripper integrator and refreshed OSC goals."""
    validate_robot_platform(scene, record)
    if record is None:
        return
    from cowm.world.env import RESET_QPOS

    robot, sim = scene.env.robots[0], scene.env.sim
    # OSC retains its orientation target for zero rotation deltas.
    orientation_goal = np.array(robot.controller.goal_ori, copy=True)
    index = {"arm_qpos": robot._ref_joint_pos_indexes,
             "grip_qpos": robot._ref_gripper_joint_pos_indexes,
             "arm_qvel": robot._ref_joint_vel_indexes,
             "grip_qvel": robot._ref_gripper_joint_vel_indexes}
    for field, ids in index.items():
        vector = np.asarray(record[field], dtype=float)
        if field.endswith("qpos"):
            sim.data.qpos[ids] = vector
        else:
            sim.data.qvel[ids] = vector
    sim.forward()
    robot.gripper.current_action = np.asarray(record["grip_current_action"], dtype=float).copy()
    robot.controller.update(force=True)
    robot.controller.update_initial_joints(RESET_QPOS)
    robot.controller.reset_goal()
    robot.controller.goal_ori = orientation_goal
