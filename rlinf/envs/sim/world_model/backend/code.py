# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code world-model backend (CoWM-PO, plan §3.9 and §9.6 S3).

``code_mode: l0_oracle`` is OracleCodeEnv-Debug: it runs the ORIGINAL LIBERO scene from privileged
init states. It exists only to test plumbing (plan §6.2) and never produces method results. Later
modes replace the scene with a Builder world (``world_vN/``) and the init state with the M1 estimate;
the session/generate contract stays the same.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

import numpy as np
import torch

from rlinf.envs.sim.world_model.registry import register_backend

from . import FrameQueue

__all__ = ["CodeBackend", "frame_diff_fraction"]


def frame_diff_fraction(a: torch.Tensor, b: torch.Tensor, px_tol: float = 0.2) -> float:
    """Fraction of pixels whose largest channel difference exceeds px_tol (0.2 in [-1, 1] = 25/255).
    Max |diff| alone is dominated by single anti-aliased edge pixels."""
    return ((a - b).abs().amax(dim=0) > px_tol).float().mean().item()


@register_backend("code")
class CodeBackend:
    """One executable world per env slot; frames come from the world's own state."""

    supports_kir = False

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.chunk = int(cfg.chunk)
        self.condition_frame_length = int(cfg.condition_frame_length)
        self.image_size = tuple(cfg.image_size)
        self.mode = cfg.get("code_mode", "l0_oracle")
        if self.mode != "l0_oracle":
            raise NotImplementedError(f"code_mode {self.mode!r} is not implemented yet")
        self.num_steps_wait = int(cfg.get("num_steps_wait", 15))
        self.check_init_frame = bool(cfg.get("check_init_frame", True))
        self.init_frame_tol = float(cfg.get("init_frame_tol", 0.005))  # max fraction of changed pixels
        with open(os.path.join(cfg.initial_image_path, "manifest.json")) as f:
            self._manifest = json.load(f)
        self._suite = None
        self._slot_episode: dict[int, str] = {}
        self._envs: dict[int, dict[str, Any]] = {}
        self._open: set[int] = set()

    @staticmethod
    def load_reward_model(cfg):
        # Same reward model and loader as the Wan backend, so both branches are scored identically.
        from diffsynth.models.reward_model import ResnetRewModel, TaskEmbedResnetRewModel

        if cfg.reward_model.type == "ResnetRewModel":
            return ResnetRewModel(cfg.reward_model.from_pretrained)
        if cfg.reward_model.type == "TaskEmbedResnetRewModel":
            return TaskEmbedResnetRewModel(
                checkpoint_path=cfg.reward_model.from_pretrained,
                task_suite_name=cfg.task_suite_name,
            )
        raise ValueError(f"Unknown reward model type: {cfg.reward_model.type}")

    def reward_instructions(self, env) -> Optional[list[str]]:
        if env.cfg.reward_model.type != "TaskEmbedResnetRewModel":
            return None
        instructions = []
        for env_idx in range(env.num_envs):
            instructions.extend([env.task_descriptions[env_idx]] * self.chunk)
        return instructions

    def set_session_episodes(self, env_ids: Sequence[int], episode_files: Sequence[str]) -> None:
        """Called by WorldModelEnv.reset before open_session: which episode each slot starts."""
        for env_id, path in zip(env_ids, episode_files):
            self._slot_episode[int(env_id)] = os.path.basename(path)

    def _benchmark(self):
        if self._suite is None:
            from libero.libero import benchmark

            self._suite = benchmark.get_benchmark_dict()[self._manifest["suite"]]()
        return self._suite

    def _env_for(self, slot: int, task_id: int):
        cur = self._envs.get(slot)
        if cur is not None and cur["task_id"] == task_id:
            return cur["env"]
        if cur is not None:
            cur["env"].close()
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task = self._benchmark().get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self.image_size[0],
            camera_widths=self.image_size[1],
            ignore_done=True,
        )
        self._envs[slot] = {"task_id": task_id, "env": env}
        return env

    @staticmethod
    def frame_from_obs(obs) -> torch.Tensor:
        """agentview, rotated 180 deg like rlinf/envs/sim/libero/utils.py, as [3, H, W] in [-1, 1]."""
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.float32) / 255.0
        return torch.from_numpy(img * 2.0 - 1.0).permute(2, 0, 1).contiguous()

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        seeds: Sequence[int],
    ) -> None:
        warm = np.zeros(7)
        warm[6] = -1.0
        for row, env_id in enumerate(env_ids):
            slot = int(env_id)
            name = self._slot_episode.get(slot)
            if name is None:
                raise RuntimeError(
                    f"slot {slot} has no episode identity; WorldModelEnv must call set_session_episodes"
                )
            ep = self._manifest["episodes"][name]
            env = self._env_for(slot, ep["task_id"])
            obs = self.reset_to_episode(env, ep, self._benchmark(), self.cfg.initial_image_path,
                                        self.num_steps_wait)
            if self.check_init_frame:
                got = self.frame_from_obs(obs)
                ref = init_frames[row][-1][:, 0].detach().float().cpu()
                frac = frame_diff_fraction(got, ref)
                if frac > self.init_frame_tol:
                    raise RuntimeError(
                        f"slot {slot} ({name}): {100 * frac:.3f}% of pixels differ from the condition frame"
                    )
            self._open.add(slot)

    @staticmethod
    def reset_to_episode(env, ep, suite, init_dir, num_steps_wait):
        """reset -> restore privileged body poses -> set_init_state -> warmup (RLinf LiberoEnv semantics)."""
        env.reset()
        priv = np.load(os.path.join(init_dir, ep["privileged"]))
        model = env.env.sim.model
        model.body_pos[:] = priv["body_pos"]
        model.body_quat[:] = priv["body_quat"]
        obs = env.set_init_state(suite.get_task_init_states(ep["task_id"])[ep["trial_id"]])
        warm = np.zeros(7)
        warm[6] = -1.0
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(warm)
        return obs

    def generate(self, env_ids: Sequence[int], actions: torch.Tensor) -> torch.Tensor:
        a = (
            actions.detach().float().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions, dtype=np.float32)
        )
        if a.shape[0] != len(env_ids):
            raise ValueError(f"env_ids and actions disagree: {len(env_ids)} vs {a.shape[0]}")
        videos = []
        for row, env_id in enumerate(env_ids):
            slot = int(env_id)
            if slot not in self._open:
                raise RuntimeError(f"slot {slot} has no open session")
            env = self._envs[slot]["env"]
            frames = []
            # actions are env-ready (prepare_actions_for_libero already applied), one low-level step each
            for k in range(a.shape[1]):
                obs, _, _, _ = env.step(a[row, k])
                frames.append(self.frame_from_obs(obs))
            videos.append(torch.stack(frames, dim=1))  # [3, T, H, W]
        return torch.stack(videos, dim=0)

    def close_session(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._open.discard(int(env_id))

    def offload(self) -> None:
        pass

    def onload(self) -> None:
        pass
