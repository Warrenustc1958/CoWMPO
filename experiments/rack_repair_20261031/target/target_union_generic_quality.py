"""Target-private quality of the generalised robot union (v6) vs the existing whole-robot track. GT only scores."""
import json
from pathlib import Path

import numpy as np

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
T = Path("/data_cf/cowmpo/target_eval/codex_s56_20261030_1020")
SRC = R / "rack_robot_union_generic_v6"
GT = {"multidir": T / "rack_multidir_dev_v1", "touch": T / "rack_touch_dev_v1"}
EP = "libero_goal_t0_i0"


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


fr = json.loads((SRC / "freeze.json").read_text())
rows = []
for rec in fr["queries"]:
    old_v, new_v = [], []
    for s in rec["steps"]:
        f = f"{EP}__f{s:03d}"
        with np.load(GT[rec["group"]] / rec["key"] / f / "gt_masks.npz") as a:
            gt = a["robot"]
        with np.load(Path(rec["old"]) / f / "masks.npz") as a:
            old = a["robot"]
        with np.load(SRC / "frames_merged" / rec["key"] / f / "masks.npz") as a:
            new = a["robot"]
        if s == 0 and not np.array_equal(old, new):
            print("NOTE initial frame differs", rec["key"])
        old_v.append(iou(gt, old))
        new_v.append(iou(gt, new))
    rows.append({"group": rec["group"], "key": rec["key"], "frames": len(new_v),
                 "old_mean": round(float(np.mean(old_v)), 4), "new_mean": round(float(np.mean(new_v)), 4),
                 "old_min": round(min(old_v), 4), "new_min": round(min(new_v), 4),
                 "old_ge08": sum(v >= .8 for v in old_v), "new_ge08": sum(v >= .8 for v in new_v),
                 "frames_worse_by_0.02": sum(n < o - .02 for o, n in zip(old_v, new_v))})
(T / "robot_union_generic_quality_v6.json").write_text(json.dumps({"scope": "Target-private dev audit", "queries": rows}, indent=1))
for r in rows:
    print(json.dumps(r))
