"""S6 Target side: execute a Builder-planned probe open-loop in the target platform.

Builder -> Target: probe.json (actions only). Target -> Builder: RGB frames at the requested steps
(evidence/<set>/<episode>__f<k>/frame0.png). Privileged GT (probe-object displacement) goes to a
Target-only directory for evaluation and is never exported to the Builder.
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

from cowm.init.gt_dump import NAME_MAP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-dir", required=True)
    p.add_argument("--probes", required=True, help="Builder probe root (<episode>/probe.json)")
    p.add_argument("--evidence-out", required=True)
    p.add_argument("--gt-out", required=True)
    a = p.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    man = json.load(open(os.path.join(a.init_dir, "manifest.json")))
    suite = benchmark.get_benchmark_dict()[man["suite"]]()
    os.makedirs(a.gt_out, exist_ok=True)
    envs, warm = {}, np.array([0, 0, 0, 0, 0, 0, -1.0])
    for ep in sorted(os.listdir(a.probes)):
        pr = json.load(open(os.path.join(a.probes, ep, "probe.json")))
        info = man["episodes"][ep + ".npy"]
        if info["task_id"] not in envs:
            task = suite.get_task(info["task_id"])
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            envs[info["task_id"]] = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                                       camera_widths=256, ignore_done=True)
        env = envs[info["task_id"]]
        env.reset()
        priv = np.load(os.path.join(a.init_dir, info["privileged"]))
        sim = env.env.sim
        sim.model.body_pos[:] = priv["body_pos"]
        sim.model.body_quat[:] = priv["body_quat"]
        obs = env.set_init_state(suite.get_task_init_states(info["task_id"])[info["trial_id"]])
        for _ in range(man["num_steps_wait"]):
            obs, _, _, _ = env.step(warm)
        m = sim.model
        root = next(m.body_id2name(b) for b in range(m.nbody)
                    if int(m.body_rootid[b]) == b and (m.body_id2name(b) or "").startswith(NAME_MAP[pr["obj"]] + "_"))
        bid = m.body_name2id(root)
        p0 = sim.data.body_xpos[bid].copy()
        frames = set(pr["frames"])

        def save(k, o):
            d = os.path.join(a.evidence_out, f"{ep}__f{k:03d}")
            os.makedirs(d, exist_ok=True)
            Image.fromarray(np.ascontiguousarray(o["agentview_image"][::-1, ::-1])).save(os.path.join(d, "frame0.png"))
            with open(os.path.join(d, "meta.json"), "w") as fh:
                json.dump({"episode": ep, "step": k, "language": info["language"]}, fh)

        if 0 in frames:
            save(0, obs)
        for k, act in enumerate(pr["actions"], start=1):
            obs, _, _, _ = env.step(np.asarray(act))
            if k in frames:
                save(k, obs)
        p1 = sim.data.body_xpos[bid].copy()
        with open(os.path.join(a.gt_out, ep + "_probe.json"), "w") as fh:
            json.dump({"obj": pr["obj"], "disp": (p1 - p0).tolist()}, fh)
        print(ep, "gt disp cm", np.round((p1 - p0)[:2] * 100, 1), flush=True)
    print("PROBE_TARGET_DONE")


if __name__ == "__main__":
    main()
