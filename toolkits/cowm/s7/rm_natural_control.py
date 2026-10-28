"""S7 RM natural positive/negative control (Target platform; read-only w.r.t. RM, thresholds, preprocessing).

Roll out the frozen OpenVLA-OFT SFT policy (the GRPO starting point) in the official LIBERO-Goal simulator on
fit-split trials, record the agentview frame every chunk with the official success flag, and score every
frame with the frozen shared RM under the deployed contract (RGB, rot180 = get_libero_image, [-1, 1]).
Diagnostic orientations are scored too but never used to change anything.

Outputs (Target-side diagnostics): <out>/episodes/<task>_<trial>.npz, <out>/report.json
  frame level  : TP/FP/FN/TN with label = official _check_success() at that step
  episode level: success (any step) vs RM fired (any frame p > 0.5); first success step vs first RM trigger
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
sys.path.insert(0, "/mnt/nfs/fjj/Code/CoWMPO")

import numpy as np
import torch
from omegaconf import OmegaConf

MODEL = "/mnt/nfs/fjj/Models/Openvla-oft-SFT-libero-goal-traj1"
RM = "/mnt/nfs/fjj/Models/RLinf-Wan-LIBERO-Goal/taskemb_resnet_rm.pth"
MODEL_YAML = "/mnt/nfs/fjj/Code/CoWMPO/examples/embodiment/config/model/openvla_oft.yaml"
ORIENT = {"contract_rot180": lambda x: x, "raw": lambda x: x[::-1, ::-1], "hflip": lambda x: x[:, ::-1],
          "vflip": lambda x: x[::-1]}   # applied on top of the contract (rot180) frame


def load_policy(dev):
    from rlinf.models.embodiment.openvla_oft import get_model

    cfg = OmegaConf.load(MODEL_YAML)
    cfg.model_path = MODEL
    cfg.unnorm_key = "libero_goal_no_noops"
    cfg.max_prompt_length = 128
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(dev).eval()
    return model


def make_env(task):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                        getattr(task, "bddl_file", None) or task.bddl_file_name)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    return env


def obs_to_state(obs):
    from rlinf.envs.sim.libero.utils import quat2axisangle

    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


def rollout_task(model, suite, task_id, trials, max_steps, temperature, post_steps, dev):
    from rlinf.envs.action_utils import prepare_actions_for_libero

    task = suite.get_task(task_id)
    inits = suite.get_task_init_states(task_id)
    envs = [make_env(task) for _ in trials]
    obs = []
    for env, tr in zip(envs, trials):
        env.reset()
        env.set_init_state(inits[tr])
        for _ in range(15):
            o, _, _, _ = env.step(np.zeros(7))
        obs.append(o)
    n = len(envs)
    frames = [[np.ascontiguousarray(o["agentview_image"][::-1, ::-1])] for o in obs]
    steps = [[0] for _ in range(n)]
    labels = [[bool(e.env._check_success())] for e in envs]
    first_succ = [None] * n
    stop_at = [max_steps] * n
    t = 0
    while t < max(stop_at):
        active = [i for i in range(n) if t < stop_at[i]]
        env_obs = {"main_images": torch.from_numpy(np.stack([frames[i][-1] for i in active])),
                   "wrist_images": torch.from_numpy(np.stack(
                       [np.ascontiguousarray(obs[i]["robot0_eye_in_hand_image"][::-1, ::-1]) for i in active])),
                   "states": torch.from_numpy(np.stack([obs_to_state(obs[i]) for i in active])),
                   "task_descriptions": [task.language] * len(active)}
        out = model.predict_action_batch(env_obs=env_obs, do_sample=True, temperature=temperature, top_k=-1,
                                         top_p=1.0, calculate_logprobs=False, calculate_values=False)
        acts = out[0] if isinstance(out, tuple) else out
        acts = acts.detach().float().cpu().numpy() if torch.is_tensor(acts) else np.asarray(acts, dtype=np.float32)
        acts = prepare_actions_for_libero(acts.reshape(len(active), -1, 7).copy(), "openvla_oft")
        for j, i in enumerate(active):
            for a in acts[j]:
                obs[i], _, _, _ = envs[i].step(a)
                if envs[i].env._check_success() and first_succ[i] is None:
                    first_succ[i] = t + 1
                    stop_at[i] = min(max_steps, t + 1 + post_steps)
            frames[i].append(np.ascontiguousarray(obs[i]["agentview_image"][::-1, ::-1]))
            steps[i].append(t + acts.shape[1])
            labels[i].append(bool(envs[i].env._check_success()))
        t += acts.shape[1]
    for e in envs:
        e.close()
    return task.language, [{"trial": tr, "frames": np.stack(frames[i]), "steps": np.array(steps[i]),
                            "labels": np.array(labels[i]), "first_success": first_succ[i]}
                           for i, tr in enumerate(trials)]


def score(rm, frames, instr, dev):
    from diffsynth.models.reward_model import instruction_to_task_id

    out = {}
    for name, fn in ORIENT.items():
        ps = []
        for k in range(0, len(frames), 128):
            x = np.stack([np.ascontiguousarray(fn(f)) for f in frames[k:k + 128]]).astype(np.float32) / 127.5 - 1
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
            with torch.no_grad():
                tid = instruction_to_task_id([instr] * len(x), "libero_goal").to(dev)
                f = torch.cat([rm.visual_encoder(x), rm.task_embedding(tid)], 1)
                ps.append(rm.fusion_layer(f).squeeze(1).float().cpu().numpy())
        out[name] = np.concatenate(ps)
    return out


def summarize(records):
    rep = {}
    for name in ORIENT:
        lab = np.concatenate([r["labels"] for r in records])
        p = np.concatenate([r["probs"][name] for r in records])
        pred = p > 0.5
        ep_s = np.array([r["first_success"] is not None for r in records])
        ep_f = np.array([bool((r["probs"][name] > 0.5).any()) for r in records])
        lag = [int(r["steps"][np.argmax(r["probs"][name] > 0.5)]) - r["first_success"] for r in records
               if r["first_success"] is not None and (r["probs"][name] > 0.5).any()]
        rep[name] = {
            "frame": {"TP": int((pred & lab).sum()), "FP": int((pred & ~lab).sum()),
                      "FN": int((~pred & lab).sum()), "TN": int((~pred & ~lab).sum()),
                      "pos_prob_median": float(np.median(p[lab])) if lab.any() else None,
                      "neg_prob_p99": float(np.quantile(p[~lab], 0.99)) if (~lab).any() else None},
            "episode": {"TP": int((ep_f & ep_s).sum()), "FP": int((ep_f & ~ep_s).sum()),
                        "FN": int((~ep_f & ep_s).sum()), "TN": int((~ep_f & ~ep_s).sum())},
            "trigger_minus_success_steps": {"n": len(lag), "median": float(np.median(lag)) if lag else None,
                                            "min": int(min(lag)) if lag else None,
                                            "max": int(max(lag)) if lag else None}}
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--trials", default="0,1,2,3,4,5,6,7,8,9", help="fit-split trial ids only (0-19)")
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--post-steps", type=int, default=32, help="keep stepping after first success")
    p.add_argument("--temperature", type=float, default=1.6)
    a = p.parse_args()
    trials = [int(x) for x in a.trials.split(",")]
    assert all(0 <= t < 20 for t in trials), "fit split only"
    os.makedirs(os.path.join(a.out, "episodes"), exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)
    from diffsynth.models.reward_model import TaskEmbedResnetRewModel
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    model = load_policy(dev)
    rm = TaskEmbedResnetRewModel(checkpoint_path=RM, task_suite_name="libero_goal").eval().to(dev)
    records, per_task = [], {}
    for tid in [int(x) for x in a.tasks.split(",")]:
        t0 = time.time()
        instr, eps = rollout_task(model, suite, tid, trials, a.max_steps, a.temperature, a.post_steps, dev)
        for e in eps:
            e["probs"] = score(rm, e["frames"], instr, dev)
            e["task_id"], e["instruction"] = tid, instr
            np.savez_compressed(os.path.join(a.out, "episodes", f"t{tid}_i{e['trial']}.npz"),
                                frames=e["frames"], steps=e["steps"], labels=e["labels"],
                                **{f"p_{k}": v for k, v in e["probs"].items()})
            records.append(e)
        ts = [r for r in records if r["task_id"] == tid]
        per_task[instr] = summarize(ts)["contract_rot180"]
        per_task[instr]["success_rate"] = float(np.mean([r["first_success"] is not None for r in ts]))
        print(f"task {tid} '{instr}' {time.time() - t0:.0f}s success={per_task[instr]['success_rate']:.2f} "
              f"episode={per_task[instr]['episode']} frame={per_task[instr]['frame']}", flush=True)
        rep = {"policy": MODEL, "rm": RM, "trials": trials, "temperature": a.temperature,
               "n_episodes": len(records), "overall": summarize(records), "per_task_contract": per_task,
               "episodes": [{"task_id": r["task_id"], "trial": r["trial"], "first_success": r["first_success"],
                             "max_p_contract": float(r["probs"]["contract_rot180"].max())} for r in records]}
        json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print("RM_NATURAL_DONE", json.dumps(rep["overall"]["contract_rot180"]), flush=True)


if __name__ == "__main__":
    main()
