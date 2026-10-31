# Copyright 2026 The RLinf Authors.
"""Contact-sensitive, Builder-visible signal for ranking rack candidates (v14).

Full-image robot-mask IoU is diluted by the arm; the lowest mask row is a fixed mount edge (v13: degenerate).
Here the robot masks are compared only inside an ROI around the rack: the bounding box of the frame-0
perception rack mask of the probe episode, expanded by PAD px. Observed = frozen v7b public tracks; predicted
= candidate world's rendered robot mask. Frames from 20 steps before the contact phase to its end. No Target.
Writes one json per candidate (incremental).
usage: python eval_rack_roi_v14.py
"""
import json
from pathlib import Path

import numpy as np

from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
V12 = R / "rack_back_dev_v12"
OUT = V12 / "roi_v14"
EP = "libero_goal_t0_i0"
PAD = 20


def roi():
    with np.load(R / "visual_dev_bound_v1/dev3" / EP / "masks.npz") as a:
        m = a["rack"]
    ys, xs = np.nonzero(m)
    return max(0, ys.min() - PAD), min(256, ys.max() + 1 + PAD), max(0, xs.min() - PAD), min(256, xs.max() + 1 + PAD)


def main():
    OUT.mkdir(exist_ok=True)
    y0, y1, x0, x1 = roi()
    qs = json.loads((V12 / "freeze.json").read_text())["queries"]
    robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    for cand in sorted(V12.glob("cand*/result.json")):
        dest = OUT / f"{cand.parent.name}.json"
        if dest.exists():
            continue
        res = json.loads(cand.read_text())
        sc = Scene(res["world"])
        doc = json.loads((Path(res["states"]) / EP / "state.json").read_text())
        per, allv = {}, []
        for q in qs:
            probe = json.loads(Path(q["source"]).read_text())
            ph = list(probe["phases"])
            i = ph.index("contact")
            start, end = probe["phases"][ph[i - 1]] - 20, probe["phases"]["contact"]
            frames = [s for s in probe["frames"] if start < s <= end]
            sc.reset()
            apply_state(sc, doc["objects"], require_matched=True, robot=robot0)
            v = []
            for step, act in enumerate(probe["actions"], 1):
                sc.env.step(np.asarray(act))
                if step in frames:
                    with np.load(Path(q["video"]) / f"{EP}__f{step:03d}" / "masks.npz") as a:
                        o = a["robot"][y0:y1, x0:x1]
                    p = sc.robot_mask()[y0:y1, x0:x1]
                    u = (o | p).sum()
                    v.append(float((o & p).sum() / u) if u else 1.0)
                if step >= end:
                    break
            per[q["key"]] = round(float(np.mean(v)), 4)
            allv += v
        sc.env.close()
        row = {"cand": cand.parent.name, "e": res["e"], "h": res["h"], "builder_loss": round(res["loss"], 4),
               "roi": [int(y0), int(y1), int(x0), int(x1)], "roi_iou_mean": round(float(np.mean(allv)), 4),
               "per_probe": per}
        dest.write_text(json.dumps(row, indent=1))
        print(json.dumps(row), flush=True)
    print("ROI_V14_DONE", flush=True)


if __name__ == "__main__":
    main()
