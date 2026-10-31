"""Target-private quality of the v7 re-detection repair vs its input track, for all 11 rack probes. GT scores only."""
import json
from pathlib import Path

import numpy as np

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
T = Path("/data_cf/cowmpo/target_eval/codex_s56_20261030_1020")
SRC = R / "rack_robot_redetect_v7"
GT = {"multidir": "rack_multidir_dev_v1", "touch": "rack_touch_dev_v1", "horizontal": "rack_isolated_dev_v2",
      "vertical": "rack_vertical_dev_v3"}
EP = "libero_goal_t0_i0"


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


rows = []
for q in json.loads((SRC / "freeze.json").read_text())["queries"]:
    ov, nv, fp_old, fp_new = [], [], 0, 0
    for s in q["steps"]:
        f = f"{EP}__f{s:03d}"
        with np.load(T / GT[q["group"]] / q["key"] / f / "gt_masks.npz") as a:
            gt = a["robot"]
        with np.load(Path(q["track"]) / f / "masks.npz") as a:
            old = a["robot"]
        with np.load(SRC / q["key"] / f / "masks.npz") as a:
            new = a["robot"]
        ov.append(iou(gt, old))
        nv.append(iou(gt, new))
        fp_old += int((old & ~gt).sum())
        fp_new += int((new & ~gt).sum())
    rows.append({"group": q["group"], "key": q["key"], "frames": len(nv),
                 "in_mean": round(float(np.mean(ov)), 4), "out_mean": round(float(np.mean(nv)), 4),
                 "in_min": round(min(ov), 4), "out_min": round(min(nv), 4),
                 "in_ge08": sum(v >= .8 for v in ov), "out_ge08": sum(v >= .8 for v in nv),
                 "frames_worse_by_0.02": sum(n < o - .02 for o, n in zip(ov, nv)),
                 "fp_px_in": fp_old, "fp_px_out": fp_new})
(T / "robot_redetect_quality_v7.json").write_text(json.dumps({"scope": "Target-private dev audit", "queries": rows},
                                                              indent=1))
tot = {k: sum(r[k] for r in rows) for k in ("frames", "in_ge08", "out_ge08", "frames_worse_by_0.02")}
for r in rows:
    print(json.dumps(r))
print("TOTAL", json.dumps(tot))
