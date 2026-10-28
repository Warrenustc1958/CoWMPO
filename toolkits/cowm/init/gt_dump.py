"""S5 Target side (NOT Builder-visible): ground truth for evaluating EvidenceAdapter + InitialStateEstimate.

For each dev episode: same reset as RLinf LiberoEnv / L0 backend (reset -> privileged fixture body poses ->
set_init_state -> 15 warmup steps), then dump object root-body poses relative to the robot base and the
agentview per-object segmentation masks (policy-image orientation). Output goes to a directory that is
not on the Builder sandbox allow-list.
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

# world-program object name -> LIBERO root-body name prefix (evaluation mapping only, Target side)
NAME_MAP = {"cabinet": "wooden_cabinet", "stove": "flat_stove", "rack": "wine_rack", "plate": "plate",
            "bowl": "akita_black_bowl", "cream_cheese": "cream_cheese", "wine_bottle": "wine_bottle"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default="")
    a = p.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    with open(os.path.join(a.init_dir, "manifest.json")) as f:
        man = json.load(f)
    suite = benchmark.get_benchmark_dict()[man["suite"]]()
    eps = [e + ".npy" if not e.endswith(".npy") else e for e in a.episodes.split(",")] if a.episodes \
        else sorted(man["episodes"])
    os.makedirs(a.out, exist_ok=True)
    envs = {}
    warm = np.zeros(7)
    warm[6] = -1.0
    for name in eps:
        ep = man["episodes"][name]
        if ep["task_id"] not in envs:
            task = suite.get_task(ep["task_id"])
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            envs[ep["task_id"]] = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                                     ignore_done=True)
        env = envs[ep["task_id"]]
        env.reset()
        priv = np.load(os.path.join(a.init_dir, ep["privileged"]))
        sim = env.env.sim
        sim.model.body_pos[:] = priv["body_pos"]
        sim.model.body_quat[:] = priv["body_quat"]
        env.set_init_state(suite.get_task_init_states(ep["task_id"])[ep["trial_id"]])
        for _ in range(man["num_steps_wait"]):
            env.step(warm)
        m, d = sim.model, sim.data
        base = d.body_xpos[m.body_name2id("robot0_base")].copy()
        roots = {}
        for b in range(m.nbody):
            bn = m.body_id2name(b) or ""
            if int(m.body_rootid[b]) == b and b != 0:
                roots[bn] = b
        obj = {}
        for ours, pre in NAME_MAP.items():
            cand = sorted(n for n in roots if n.startswith(pre + "_"))
            if not cand:
                continue
            b = roots[cand[0]]
            q = d.body_xquat[b]
            yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
            obj[ours] = {"body": cand[0], "pos_rel_base": (d.body_xpos[b] - base).tolist(), "yaw": yaw, "bid": b}
        s = sim.render(width=256, height=256, camera_name="agentview", segmentation=True)
        gid = s[..., 1].astype(np.int64)
        gid[(s[..., 0] != 5) | (gid < 0)] = -1
        gid = gid[::-1, ::-1]
        groot = np.where(gid >= 0, m.body_rootid[m.geom_bodyid[np.clip(gid, 0, None)]], -1)
        masks = {k: groot == v["bid"] for k, v in obj.items()}
        stem = name[:-4]
        np.savez_compressed(os.path.join(a.out, stem + "_masks.npz"), **masks)
        with open(os.path.join(a.out, stem + ".json"), "w") as fh:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "bid"} for k, v in obj.items()}, fh, indent=1)
        print(stem, {k: [round(x, 3) for x in v["pos_rel_base"][:2]] for k, v in obj.items()},
              {k: int(v.sum()) for k, v in masks.items()}, flush=True)
    print("GT_DONE", len(eps))


if __name__ == "__main__":
    main()
