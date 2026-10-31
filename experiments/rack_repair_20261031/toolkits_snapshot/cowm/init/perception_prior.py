"""S5 perception prior (Builder side): what the world program itself predicts the first frame looks like.

Renders the compiled world at its nominal poses after the 15-step warmup and saves
  robot_mask  : pixels covered by the robot (shared platform prior; boxes mostly on it are rejected)
  per object  : expected mask area and box size in pixels (used only as a loose plausibility filter)
No target information is used.
"""
import argparse
import json
import os

import numpy as np

from cowm.init.estimate_state import Scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True)
    p.add_argument("--out", required=True, help="output dir (perception_prior.npz / .json)")
    a = p.parse_args()
    sc = Scene(a.world)
    sc.reset()
    seg, rob = sc.seg(), sc.robot_mask()
    objs = {}
    for k, n in enumerate(sc.names):
        ys, xs = np.nonzero(seg == k)
        o = sc.world["objects"][k]
        objs[n] = {"area": int(len(xs)), "category": o["category"], "material": o.get("material"),
                   "material_class": sc.world["materials"].get(o.get("material"), None),
                   "box_wh": [int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1] if len(xs) else None}
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "perception_prior.npz"), robot_mask=rob)
    json.dump({"world_sha256": sc.world["world_sha256"], "objects": objs},
              open(os.path.join(a.out, "perception_prior.json"), "w"), indent=1)
    print("PRIOR", {n: v["area"] for n, v in objs.items()}, "robot px", int(rob.sum()))


if __name__ == "__main__":
    main()
