"""Write the frozen experiment profile (goal_v1) and the fit/dev/gate/test split manifest.

Run on the Target side (reads model files for hashes; does not read LIBERO scene internals).
Split rule (plan §3.8D, §9.7), per task, by init/trial id (the unit an episode is built from):
  fit  0-19   world parameter fitting / probe trajectories
  dev  20-29  counterexamples the Agent may see
  gate 30-39  accept/reject only, Agent never sees samples, query budget per world version
  test 40-49  sealed until the method is frozen (policy evaluation on these trials only)
Episodes that were ever shown to the Agent are recorded as 'exposed' and count as dev forever.
"""
import argparse
import hashlib
import json
import os


def sha(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read() if limit is None else f.read(limit))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--exposed-evidence", default="/data_cf/cowmpo/evidence/libero_goal_dev3")
    a = p.parse_args()
    M = "/mnt/nfs/fjj/Models"
    profile = {
        "name": "goal_v1", "frozen": "2026-10-28",
        "suite": "libero_goal", "n_tasks": 10, "trials_per_task": 50,
        "policy": {"repo": "Haozhan72/Openvla-oft-SFT-libero-goal-traj1",
                   "revision": "d20e1d447dfd87c0daa121b0739e2a379f7fe334",
                   "path": f"{M}/Openvla-oft-SFT-libero-goal-traj1",
                   "index_sha256": sha(f"{M}/Openvla-oft-SFT-libero-goal-traj1/model.safetensors.index.json"),
                   "unnorm_key": "libero_goal_no_noops"},
        "wan": {"repo": "RLinf/RLinf-Wan-LIBERO-Goal", "revision": "bd395971c3467de3dd19e7e6c7562af48a2894a6",
                "path": f"{M}/RLinf-Wan-LIBERO-Goal"},
        "reward_model": {"type": "TaskEmbedResnetRewModel",
                         "path": f"{M}/RLinf-Wan-LIBERO-Goal/taskemb_resnet_rm.pth",
                         "sha256": sha(f"{M}/RLinf-Wan-LIBERO-Goal/taskemb_resnet_rm.pth"),
                         "frozen": True, "note": "shared by Wan and Code branches; repair Agent may not change it"},
        "env": {"controller": "OSC_POSE", "control_freq": 20, "physics_dt": 0.002, "num_steps_wait": 15,
                "reset_gripper_open": True, "image": [256, 256], "image_flip": "rotate 180 deg",
                "action": "7-D env-ready after prepare_actions_for_libero", "chunk": 8,
                "max_episode_steps": 512},
        "success": {"primary": "success_once (any step), RLinf eval",
                    "secondary": "success_at_end", "eval_sampling": "do_sample, temperature 1.6"},
        "baseline_S0": {"success_once": 0.504, "success_at_end": 0.288, "n": 500,
                        "note": "all 50 trials; test-split SR must be recomputed on trials 40-49"},
        "rlinf_commit": "e1a82ebf0c1af69da6ab9b1519f5f3692fe33566",
    }
    exposed = sorted(d for d in os.listdir(a.exposed_evidence)) if os.path.isdir(a.exposed_evidence) else []
    roles = {"fit": list(range(0, 20)), "dev": list(range(20, 30)), "gate": list(range(30, 40)),
             "test": list(range(40, 50))}
    splits = {"profile": "goal_v1", "unit": "(task_id, trial_id)", "roles": roles,
              "exposed_to_agent": exposed,
              "rules": ["an episode shown to the Agent (prompt, image, report) is dev forever",
                        "gate returns only accept/reject + aggregates; max 5 gate queries per world version",
                        "test is sealed until the method (SDK, verifier, thresholds) is frozen"]}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(profile, open(os.path.join(a.out_dir, "profile_goal_v1.json"), "w"), indent=2)
    json.dump(splits, open(os.path.join(a.out_dir, "splits_goal_v1.json"), "w"), indent=2)
    print("wrote", a.out_dir, "exposed:", len(exposed))


if __name__ == "__main__":
    main()
