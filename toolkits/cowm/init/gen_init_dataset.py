"""S3: build a LIBERO init dataset in the Wan npy format, plus manifest.json (file -> task, trial).

Each episode file holds the policy's first observation, produced exactly as RLinf LiberoEnv.reset does:
reset -> set_init_state -> num_steps_wait zero steps (gripper -1) -> agentview rotated 180 deg
(rlinf/envs/sim/libero/utils.py). Target side only: imports libero.
"""
import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--tasks", default="0-9", help="inclusive range, e.g. 0-9")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--num-steps-wait", type=int, default=15)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    os.makedirs(os.path.join(args.out, "privileged"), exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    lo, hi = (int(x) for x in args.tasks.split("-"))
    manifest = {"suite": args.suite, "num_steps_wait": args.num_steps_wait, "size": args.size,
                "privileged": "init states from LIBERO; L0 plumbing only (plan §6.2)", "episodes": {}}
    warm = np.zeros(7)
    warm[6] = -1.0
    for t in range(lo, hi + 1):
        task = suite.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(t)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.size, camera_widths=args.size)
        for i in range(min(args.trials, len(init_states))):
            env.reset()
            obs = env.set_init_state(init_states[i])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(warm)
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
            name = f"{args.suite}_t{t}_i{i}.npy"
            frame = {"image": img, "reward": 0.0, "instruction": task.language,
                     "delta_action": np.zeros(7, dtype=np.float32)}
            arr = np.empty(1, dtype=object)
            arr[0] = frame
            np.save(os.path.join(args.out, name), arr, allow_pickle=True)
            # LIBERO re-samples fixture poses (cabinet, stove, rack) into model.body_pos on every reset;
            # init_states only hold qpos/qvel. Save the exact body poses so L0 can rebuild this scene.
            m = env.env.sim.model
            priv = os.path.join("privileged", name.replace(".npy", ".npz"))
            np.savez(os.path.join(args.out, priv), body_pos=m.body_pos.copy(), body_quat=m.body_quat.copy())
            manifest["episodes"][name] = {"task_id": t, "trial_id": i, "language": task.language,
                                          "privileged": priv}
        env.close()
        print(f"task {t} done: {task.language}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("GEN_DONE", len(manifest["episodes"]))


if __name__ == "__main__":
    main()
