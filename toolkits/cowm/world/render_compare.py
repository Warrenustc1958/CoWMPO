"""Render a compiled world's agentview (CPU/OSMesa) next to a target evidence frame. Builder sandbox."""
import argparse
import os

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True)
    p.add_argument("--evidence", required=True, help="evidence episode dir with frame0.png")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from cowm.world.env import CodeWorldEnv

    env = CodeWorldEnv(a.world, use_camera_obs=True, camera_names="agentview", camera_heights=256,
                       camera_widths=256, has_renderer=False, has_offscreen_renderer=True)
    hold = np.zeros(7)
    hold[6] = -1.0
    obs = env.reset()
    for _ in range(15):
        obs, _, _, _ = env.step(hold)
    ours = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # same 180 deg as RLinf LIBERO
    tgt = np.asarray(Image.open(os.path.join(a.evidence, "frame0.png")).convert("RGB"))
    Image.fromarray(np.concatenate([tgt, ours], axis=1)).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
