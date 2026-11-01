"""S6 probe planning (Builder side, plan §9.7-D): an active query the Target executes open-loop.

From the Builder's own S5 estimate (no target internals) plan a sideways push on a probe object:
close the gripper, go above the far side of the object, descend, push along -y, retreat. Actions are
OSC_POSE deltas computed closed-loop in the code world (executor parity with the target is the S2 result),
then saved so the Target replays exactly the same sequence.

Output: <out>/<episode>/probe.json {actions: [[7]...], frames: [k...], obj, push}
"""
import argparse
import json
import os

import numpy as np

from cowm.init.estimate_state import Scene

GAIN = 10.0          # action = clip(GAIN * (target - eef)); OSC output_max 0.05 m per unit
TOL = 0.004
PUSH = 0.045         # fingertip travel beyond first contact (m)
CLEAR = 0.03         # approach clearance outside the object radius (m)
FRAME_EVERY = 10


def drive(scene, target, grip, acts, max_steps=50):
    """Fixed-length P segment (an early exit at TOL stalls under OSC lag; fixed length also keeps
    probe lengths identical across episodes)."""
    for _ in range(max_steps):
        d = np.asarray(target) - scene.env.eef_pos()
        a = np.zeros(7)
        a[:3] = np.clip(GAIN * d, -1.0, 1.0)
        a[6] = grip
        scene.env.step(a)
        acts.append(a.tolist())


def apply_state(scene, state, require_matched=False, robot=None):
    """Install poses; planning and scored replay require verified identities.

    Mechanical reset/snapshot diagnostics may install arbitrary poses without identity metadata.
    """
    from cowm.init.observation import state_issues
    from cowm.init.robot_state import robot_issues, install_robot_state, validate_robot_platform

    issues = state_issues(state, [o["name"] for o in scene.world["objects"]]) if require_matched else {}
    invalid_robot = robot_issues(robot)
    if invalid_robot:
        issues["_robot"] = invalid_robot
    if issues:
        raise ValueError(f"state cannot initialize a probe: {issues}")
    validate_robot_platform(scene, robot)
    for k, o in enumerate(scene.world["objects"]):
        s = state[o["name"]]
        scene.set_pose(k, s["pos"][:2], s["yaw"], s["pos"][2])
    if robot is not None:
        install_robot_state(scene, robot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True)
    p.add_argument("--state", required=True, help="S5 output root (<episode>/state.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--obj", default="bowl")
    p.add_argument("--episodes", default="")
    p.add_argument("--mode", choices=["push", "strike"], default="push")
    p.add_argument("--strike-steps", type=int, default=4)
    p.add_argument("--hold-steps", type=int, default=30)
    a = p.parse_args()
    scene = Scene(a.world)
    k = scene.names.index(a.obj)
    o = scene.world["objects"][k]
    radius = float(o["attrs"].get("radius", 0.05))
    top = scene.world["table"]["height"]
    eps = a.episodes.split(",") if a.episodes else sorted(
        e for e in os.listdir(a.state) if os.path.isfile(os.path.join(a.state, e, "state.json")))
    for ep in eps:
        document = json.load(open(os.path.join(a.state, ep, "state.json")))
        state = document["objects"]
        scene.reset()
        if document.get("robot_required") and document.get("robot") is None:
            raise ValueError(f"{ep}: required robot initialization is missing")
        apply_state(scene, state, require_matched=True, robot=document.get("robot"))
        bx, by = state[a.obj]["pos"][:2]
        z = top + min(0.03, o["top_z"] * 0.6)
        acts = []
        for _ in range(8):                                   # close the gripper in place
            acts.append([0, 0, 0, 0, 0, 0, 1.0])
            scene.env.step(np.array(acts[-1]))
        y0, y1 = by + radius + CLEAR, by + radius - PUSH
        drive(scene, [bx, y0, z + 0.08], 1.0, acts)          # above the far (+y) side
        drive(scene, [bx, y0, z], 1.0, acts)                 # descend
        if a.mode == "push":
            drive(scene, [bx, y1, z], 1.0, acts, max_steps=60)   # slow push along -y
            strike_at = None
        else:
            # strike: close in to ~1 cm, then STRIKE_STEPS of full-speed -y, then hold still while the
            # object slides freely (slide length after release depends on table friction and mass)
            drive(scene, [bx, by + radius + 0.012, z], 1.0, acts, max_steps=40)
            strike_at = len(acts)
            for _ in range(a.strike_steps):
                acts.append([0, -1.0, 0, 0, 0, 0, 1.0])
                scene.env.step(np.array(acts[-1]))
            for _ in range(a.hold_steps):
                acts.append([0, 0, 0, 0, 0, 0, 1.0])
                scene.env.step(np.array(acts[-1]))
        drive(scene, [scene.env.eef_pos()[0], scene.env.eef_pos()[1] + 0.03, z + 0.08], 1.0, acts)  # retreat
        frames = list(range(0, len(acts) + 1, FRAME_EVERY))
        if strike_at is not None:   # dense frames around the free slide
            frames = sorted(set(frames) | set(range(strike_at, strike_at + a.strike_steps + a.hold_steps + 1, 3)))
        if frames[-1] != len(acts):
            frames.append(len(acts))
        d = os.path.join(a.out, ep)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "probe.json"), "w") as fh:
            json.dump({"episode": ep, "obj": a.obj, "actions": acts, "frames": frames, "mode": a.mode,
                       "strike_at": strike_at, "push": {"y0": y0, "y1": y1, "z": z}}, fh)
        moved = scene.env.object_pos(a.obj)[:2] - np.array([bx, by])
        print(ep, "steps", len(acts), "frames", len(frames), "pred obj displacement cm",
              np.round(moved * 100, 1), flush=True)


if __name__ == "__main__":
    main()
