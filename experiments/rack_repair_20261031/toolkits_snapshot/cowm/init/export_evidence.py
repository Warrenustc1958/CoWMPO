"""Target side: export Builder-visible evidence (plan §3.7C). Only RGB + task language leave the Target.

For each episode of an init dataset: <out>/<episode>/frame0.png (policy's first agentview frame, after the
RLinf warmup) and meta.json {episode, language}. Privileged files (init states, fixture body poses) are NOT
exported.
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = 0
    for f in sorted(glob.glob(os.path.join(a.init_dir, "*.npy"))):
        ep = os.path.basename(f)[:-4]
        frame = np.load(f, allow_pickle=True)[0]
        d = os.path.join(a.out, ep)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(np.asarray(frame["image"], dtype=np.uint8)).save(os.path.join(d, "frame0.png"))
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump({"episode": ep, "language": str(frame["instruction"])}, fh)
        n += 1
    print("EXPORTED", n, "episodes ->", a.out)


if __name__ == "__main__":
    main()
