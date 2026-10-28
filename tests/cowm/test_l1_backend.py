"""L1 backend contract: open_session from S5 state, world-rendered frame 0, generate one chunk."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
from omegaconf import OmegaConf
from rlinf.envs.sim.world_model.backend.code import CodeBackend

cfg = OmegaConf.load("examples/embodiment/config/env/code_l1_libero_goal.yaml")
cfg.initial_image_path = "/data_cf/cowmpo/init/libero_goal_dev3"
be = CodeBackend(cfg, torch.device("cpu"))
eps = ["libero_goal_t0_i0.npy", "libero_goal_t4_i1.npy"]
be.set_session_episodes([0, 1], eps)
be.open_session([0, 1], init_frames=None, init_actions=None, seeds=[0, 0])
fr = be.session_frames([0, 1])
assert fr[0].shape == (3, 256, 256) and fr[0].min() >= -1 and fr[0].max() <= 1
a = np.zeros((2, cfg.chunk, 7), np.float32); a[..., 0] = 0.5; a[..., 6] = -1
v = be.generate([0, 1], torch.from_numpy(a))
assert v.shape == (2, 3, cfg.chunk, 256, 256), v.shape
moved = (v[0, :, -1] - fr[0]).abs().amax(0).gt(0.2).float().mean().item()
print("L1_CONTRACT_OK frame0", tuple(fr[0].shape), "video", tuple(v.shape), "changed px frac", round(moved, 4))
