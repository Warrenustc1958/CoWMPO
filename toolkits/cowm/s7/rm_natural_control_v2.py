"""S7 RM natural control v2 (Target platform; read-only w.r.t. RM, thresholds, preprocessing).

Fixes over v1 (audit 10.28): first success is counted per ACTION (v1 stored the chunk start, so every
"RM lag" was 7 by construction); every executed step's frame is scored (v1 scored 1 of 8); the deployed
WorldModelEnv protocol is reconstructed (chunk of 8 frames -> chunk reward = max over its frames, episode
"RM success" = first chunk with reward > 0.5); executed actions are saved so the same actions can be replayed
in the code world (L1 open-loop line of the paired diagnosis).

Outputs: <out>/episodes/t<task>_i<trial>.npz  actions [T,7], labels [T+1], p [T+1] (frame 0 = after warmup),
         frames every --frame-every steps (+ indices); <out>/report.json
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
CHUNK = 8


def load_policy(dev):
    from rlinf.models.embodiment.openvla_oft import get_model

    cfg = OmegaConf.load(MODEL_YAML)
    cfg.model_path, cfg.unnorm_key, cfg.max_prompt_length = MODEL, "libero_goal_no_noops", 128
    return get_model(cfg, torch_dtype=torch.bfloat16).to(dev).eval()


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


def frame(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])   # get_libero_image: rot180


class RMScorer:
    def __init__(self, dev):
        from diffsynth.models.reward_model import TaskEmbedResnetRewModel

        self.dev = dev
        self.rm = TaskEmbedResnetRewModel(checkpoint_path=RM, task_suite_name="libero_goal").eval().to(dev)

    @torch.no_grad()
    def __call__(self, frames, instr):
        from diffsynth.models.reward_model import instruction_to_task_id

        x = torch.from_numpy(np.stack(frames).astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).to(self.dev)
        tid = instruction_to_task_id([instr] * len(x), "libero_goal").to(self.dev)
        f = torch.cat([self.rm.visual_encoder(x), self.rm.task_embedding(tid)], 1)
        return self.rm.fusion_layer(f).squeeze(1).float().cpu().numpy()


def rollout_task(model, rm, suite, task_id, trials, a):
    from rlinf.envs.action_utils import prepare_actions_for_libero

    task = suite.get_task(task_id)
    inits = suite.get_task_init_states(task_id)
    envs = [make_env(task) for _ in trials]
    n = len(envs)
    obs, rec = [], []
    for env, tr in zip(envs, trials):
        env.reset()
        env.set_init_state(inits[tr])
        for _ in range(15):
            o, _, _, _ = env.step(np.zeros(7))
        obs.append(o)
        f0 = frame(o)
        rec.append({"trial": tr, "actions": [], "labels": [bool(env.env._check_success())],
                    "p": [float(rm([f0], task.language)[0])], "frames": [f0], "frame_idx": [0],
                    "first_success": None})
    stop_at = [a.max_steps] * n
    t = 0
    while t < max(stop_at):
        active = [i for i in range(n) if t < stop_at[i]]
        env_obs = {"main_images": torch.from_numpy(np.stack([frame(obs[i]) for i in active])),
                   "wrist_images": torch.from_numpy(np.stack(
                       [np.ascontiguousarray(obs[i]["robot0_eye_in_hand_image"][::-1, ::-1]) for i in active])),
                   "states": torch.from_numpy(np.stack([obs_to_state(obs[i]) for i in active])),
                   "task_descriptions": [task.language] * len(active)}
        out = model.predict_action_batch(env_obs=env_obs, do_sample=True, temperature=a.temperature, top_k=-1,
                                         top_p=1.0, calculate_logprobs=False, calculate_values=False)
        acts = out[0] if isinstance(out, tuple) else out
        acts = acts.detach().float().cpu().numpy() if torch.is_tensor(acts) else np.asarray(acts, dtype=np.float32)
        acts = prepare_actions_for_libero(acts.reshape(len(active), -1, 7).copy(), "openvla_oft")
        for j, i in enumerate(active):
            r, chunk_frames = rec[i], []
            for k, act in enumerate(acts[j]):
                step = t + k + 1                      # global step index of this action (1-based)
                obs[i], _, _, _ = envs[i].step(act)
                ok = bool(envs[i].env._check_success())
                r["actions"].append(np.asarray(act, dtype=np.float32))
                r["labels"].append(ok)
                f = frame(obs[i])
                chunk_frames.append(f)
                if step % a.frame_every == 0:
                    r["frames"].append(f)
                    r["frame_idx"].append(step)
                if ok and r["first_success"] is None:
                    r["first_success"] = step
                    stop_at[i] = min(a.max_steps, step + a.post_steps)
            r["p"].extend(rm(chunk_frames, task.language).tolist())   # every executed frame is scored
        t += acts.shape[1]
    for e in envs:
        e.close()
    for r in rec:
        r.update(actions=np.stack(r["actions"]), labels=np.array(r["labels"]), p=np.array(r["p"]),
                 frames=np.stack(r["frames"]), frame_idx=np.array(r["frame_idx"]))
    return task.language, rec


def chunk_protocol(p):
    """Deployed WorldModelEnv: frames 1..T in chunks of CHUNK; chunk reward = max p over its frames."""
    q = p[1:]
    nch = len(q) // CHUNK
    cm = q[:nch * CHUNK].reshape(nch, CHUNK).max(1)
    first = int(np.argmax(cm > 0.5)) if (cm > 0.5).any() else None
    return cm, (None if first is None else (first + 1) * CHUNK)   # step at which the env would see success


def episode_metrics(r):
    p, lab, fs = r["p"], r["labels"], r["first_success"]
    pred = p > 0.5
    first_rm = int(np.argmax(pred)) if pred.any() else None
    cm, first_chunk = chunk_protocol(p)
    m = {"first_success": fs, "first_rm_frame": first_rm, "first_rm_chunk_step": first_chunk,
         "rm_fires": first_rm is not None,
         "early_false_trigger": bool(first_rm is not None and (fs is None or first_rm < fs)),
         "lag_frame": None if fs is None or first_rm is None or first_rm < fs else first_rm - fs}
    if fs is not None:   # behaviour after the official success: drop-outs while the label stays true
        post = slice(fs, len(p))
        m["post_success_frames"] = int(lab[post].sum())
        m["post_success_neg"] = int((lab[post] & ~pred[post]).sum())
        m["post_success_neg_after_fire"] = int((lab[post] & ~pred[post] &
                                                (np.arange(len(p))[post] > (first_rm if first_rm is not None
                                                                            else len(p)))).sum())
    return m


def summarize(records):
    lab = np.concatenate([r["labels"] for r in records])
    pred = np.concatenate([r["p"] for r in records]) > 0.5
    em = [episode_metrics(r) for r in records]
    succ = np.array([m["first_success"] is not None for m in em])
    fire = np.array([m["rm_fires"] for m in em])
    chunk_fire = np.array([m["first_rm_chunk_step"] is not None for m in em])
    lags = [m["lag_frame"] for m in em if m["lag_frame"] is not None]
    cnt = lambda a, b: {"TP": int((a & b).sum()), "FP": int((a & ~b).sum()), "FN": int((~a & b).sum()),
                        "TN": int((~a & ~b).sum())}
    return {"n_episodes": len(records), "n_success": int(succ.sum()),
            "frame_every_step": cnt(pred, lab),
            "episode_any_frame": cnt(fire, succ),
            "episode_chunk_protocol": cnt(chunk_fire, succ),
            "early_false_trigger": int(sum(m["early_false_trigger"] for m in em)),
            "lag_frames": {"n": len(lags), "median": float(np.median(lags)) if lags else None,
                           "p90": float(np.quantile(lags, 0.9)) if lags else None,
                           "max": int(max(lags)) if lags else None},
            "post_success_neg": int(sum(m.get("post_success_neg", 0) for m in em)),
            "post_success_neg_after_fire": int(sum(m.get("post_success_neg_after_fire", 0) for m in em)),
            "post_success_frames": int(sum(m.get("post_success_frames", 0) for m in em))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--trials", default="0,1,2", help="fit-split trial ids only (0-19)")
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--post-steps", type=int, default=64, help="keep stepping after the first success")
    p.add_argument("--frame-every", type=int, default=4, help="saved frames (all frames are scored)")
    p.add_argument("--temperature", type=float, default=1.6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    trials = [int(x) for x in a.trials.split(",")]
    assert all(0 <= t < 20 for t in trials), "fit split only"
    os.makedirs(os.path.join(a.out, "episodes"), exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    model = load_policy(dev)
    rm = RMScorer(dev)
    records, per_task = [], {}
    cfg = vars(a) | {"policy": MODEL, "rm": RM, "chunk": CHUNK, "action_post": "prepare_actions_for_libero",
                     "image": "agentview rot180 RGB [-1,1] 256px", "warmup": "15 x zeros(7)"}
    for tid in [int(x) for x in a.tasks.split(",")]:
        t0 = time.time()
        instr, eps = rollout_task(model, rm, suite, tid, trials, a)
        for e in eps:
            e["task_id"], e["instruction"] = tid, instr
            np.savez_compressed(os.path.join(a.out, "episodes", f"t{tid}_i{e['trial']}.npz"),
                                actions=e["actions"], labels=e["labels"], p=e["p"], frames=e["frames"],
                                frame_idx=e["frame_idx"], task_id=tid, trial=e["trial"], instruction=instr)
            records.append(e)
        ts = [r for r in records if r["task_id"] == tid]
        per_task[instr] = summarize(ts)
        print(f"task {tid} '{instr}' {time.time() - t0:.0f}s succ={per_task[instr]['n_success']}/{len(ts)} "
              f"ep_chunk={per_task[instr]['episode_chunk_protocol']} early={per_task[instr]['early_false_trigger']} "
              f"lag={per_task[instr]['lag_frames']} post_neg={per_task[instr]['post_success_neg']}", flush=True)
        rep = {"config": cfg, "overall": summarize(records), "per_task": per_task,
               "episodes": [dict(task_id=r["task_id"], trial=r["trial"], **episode_metrics(r)) for r in records]}
        json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print("RM_NATURAL_V2_DONE", json.dumps(rep["overall"]), flush=True)


if __name__ == "__main__":
    main()
