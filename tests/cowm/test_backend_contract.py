"""S3 gate: CodeBackend (l0_oracle) honours the WorldModelBackend contract and matches real LIBERO.

Checks
  1. frames from backend.generate == frames from an independent LIBERO env given the same actions
  2. subset reset: closing/reopening one slot does not change the other slot's trajectory
  3. WorldModelEnv(backend=code): reset -> chunk_step, shapes, finite rewards from the shared RM
Run from the repo root with the Target venv (it imports libero; L0 is privileged by design).
"""
import glob
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

CFG_PATH = os.path.join(REPO, "examples/embodiment/config/env/code_libero_goal.yaml")


def load_cfg(init_dir):
    cfg = OmegaConf.load(CFG_PATH)
    cfg.initial_image_path = init_dir
    cfg.video_cfg = {"save_video": False, "info_on_video": False, "video_base_dir": "/tmp/cowm_s3_video"}
    return cfg


def init_frames_from_npy(path, cfl):
    img = np.load(path, allow_pickle=True)[0]["image"].astype(np.float32) / 255.0
    f = torch.from_numpy(img * 2 - 1).permute(2, 0, 1)[:, None]  # [3, 1, H, W]
    return [f.clone() for _ in range(cfl)]


def reference_run(manifest_ep, actions, size, init_dir, wait=15):
    """Independent LIBERO env, same episode reset; returns frames [3,T,H,W] and qpos per step [T, nq]."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.sim.world_model.backend.code import CodeBackend

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(manifest_ep["task_id"])
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size, ignore_done=True)
    CodeBackend.reset_to_episode(env, manifest_ep, suite, init_dir, wait)
    frames, qpos = [], []
    for a in actions:
        obs, _, _, _ = env.step(a)
        frames.append(CodeBackend.frame_from_obs(obs))
        qpos.append(env.env.sim.data.qpos.copy())
    env.close()
    return torch.stack(frames, dim=1), np.stack(qpos)


def smooth_actions(rng, n):
    raw = rng.uniform(-0.4, 0.4, size=(n, 6))
    k = np.ones(4) / 4
    sm = np.stack([np.convolve(raw[:, j], k, mode="same") for j in range(6)], axis=1)
    grip = np.where(np.arange(n) < n // 2, -1.0, 1.0)[:, None]
    return np.concatenate([sm, grip], axis=1).astype(np.float32)


def main():
    init_dir = sys.argv[1]
    import json

    import rlinf.envs.sim.world_model.backend.code  # noqa: F401  (registers "code")
    from rlinf.envs.sim.world_model.registry import get_backend

    cfg = load_cfg(init_dir)
    manifest = json.load(open(os.path.join(init_dir, "manifest.json")))
    files = sorted(glob.glob(os.path.join(init_dir, "*.npy")))
    names = [os.path.basename(f) for f in files]
    pick = [names[0], names[len(names) // 2]]
    ok = True

    # 1 + 2: backend level
    B = get_backend("code")
    be = B(cfg, torch.device("cpu"))
    T, chunk = 3, be.chunk
    rng = np.random.default_rng(0)
    acts = {s: smooth_actions(rng, T * chunk) for s in (0, 1)}
    be.set_session_episodes([0, 1], pick)
    be.open_session([0, 1], [init_frames_from_npy(os.path.join(init_dir, n), be.condition_frame_length)
                             for n in pick], torch.zeros(2, be.condition_frame_length, 7), [0, 0])
    got = {0: [], 1: []}
    got_q = {0: [], 1: []}
    for c in range(T):
        a = torch.from_numpy(np.stack([acts[s][c * chunk:(c + 1) * chunk] for s in (0, 1)]))
        v = be.generate([0, 1], a)
        assert v.shape == (2, 3, chunk, *be.image_size), v.shape
        for s in (0, 1):
            got[s].append(v[s])
            got_q[s].append(be._envs[s]["env"].env.sim.data.qpos.copy())  # qpos at chunk end
        if c == 0:  # subset reset of slot 1 mid-episode; slot 0 must be unaffected
            be.close_session([1])
            be.set_session_episodes([1], [pick[1]])
            be.open_session([1], [init_frames_from_npy(os.path.join(init_dir, pick[1]), be.condition_frame_length)],
                            torch.zeros(1, be.condition_frame_length, 7), [0])
    from rlinf.envs.sim.world_model.backend.code import frame_diff_fraction

    ref0, q0 = reference_run(manifest["episodes"][pick[0]], acts[0], be.image_size[0], init_dir)
    # slot 1 restarted after chunk 0: its chunks 1..T-1 equal a fresh run of chunks 1..T-1
    ref1, q1 = reference_run(manifest["episodes"][pick[1]], acts[1][chunk:], be.image_size[0], init_dir)
    checks = [
        ("slot0 qpos == real LIBERO", np.abs(np.stack(got_q[0]) - q0[chunk - 1::chunk]).max()),
        ("slot1 qpos after subset reset", np.abs(np.stack(got_q[1][1:]) - q1[chunk - 1::chunk]).max()),
    ]
    for name, d in checks:
        good = d < 1e-9
        ok &= bool(good)
        print(f"[{'PASS' if good else 'FAIL'}] {name:36s} max|dqpos|={d:.2e}")
    for name, a_, b_ in (("slot0 frames == real LIBERO", torch.cat(got[0], dim=1), ref0),
                         ("slot1 frames after subset reset", torch.cat(got[1][1:], dim=1), ref1)):
        fr = max(frame_diff_fraction(a_[:, t], b_[:, t]) for t in range(a_.shape[1]))
        good = fr <= 0.005
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name:36s} worst frame px changed={100 * fr:.3f}%")

    # 3: WorldModelEnv plumbing with the shared RM
    from rlinf.envs.sim.world_model.env import WorldModelEnv

    cfg3 = load_cfg(init_dir)
    env = WorldModelEnv(cfg3, num_envs=2, seed_offset=0, total_num_processes=1, record_metrics=True)
    obs, _ = env.reset()
    shape_ok = tuple(obs["main_images"].shape) == (2, *be.image_size, 3)
    a = torch.from_numpy(np.stack([smooth_actions(rng, chunk) for _ in range(2)]))
    out = env.chunk_step(a)
    rew = out[1]
    rew_ok = tuple(rew.shape) == (2, chunk) and torch.isfinite(rew).all().item()
    for name, good, info in (("WorldModelEnv reset obs shape", shape_ok, tuple(obs["main_images"].shape)),
                             ("chunk_step rewards finite [2,8]", rew_ok, tuple(rew.shape))):
        ok &= bool(good)
        print(f"[{'PASS' if good else 'FAIL'}] {name:36s} {info}")
    print("S3_CONTRACT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
