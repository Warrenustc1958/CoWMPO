"""S5 evaluation (host side, reads Target GT + Builder outputs; the Builder never sees this report).

Per object: xy error of the estimated pose vs target root-body pose (both relative to the robot base;
note template origins differ from LIBERO asset origins, so a few cm of offset is irreducible),
IoU(SAM mask, GT mask) = perception quality, IoU(world render, GT mask) = state-estimate quality.
"""
import argparse
import json
import os

import numpy as np


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--est", required=True, help="estimate_state output root")
    p.add_argument("--gt", required=True, help="gt_dump output dir")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows, eps = {}, sorted(e for e in os.listdir(a.est) if os.path.isfile(os.path.join(a.est, e, "state.json")))
    for ep in eps:
        st = json.load(open(os.path.join(a.est, ep, "state.json")))["objects"]
        gt = json.load(open(os.path.join(a.gt, ep + ".json")))
        gm = np.load(os.path.join(a.gt, ep + "_masks.npz"))
        sm = np.load(os.path.join(a.est, ep, "masks.npz"))
        rm = np.load(os.path.join(a.est, ep, "render_masks.npz"))
        for n, g in gt.items():
            r = rows.setdefault(n, {"err_cm": [], "sam_iou": [], "render_iou": [], "detected": 0, "n": 0})
            r["n"] += 1
            e = np.array(st[n]["pos_rel_base"][:2]) - np.array(g["pos_rel_base"][:2])
            r["err_cm"].append(float(np.linalg.norm(e) * 100))
            if n in sm.files and not st[n].get("mask_rejected"):
                r["detected"] += 1
                r["sam_iou"].append(iou(sm[n], gm[n]))
            r["render_iou"].append(iou(rm[n], gm[n]))
    summ = {}
    for n, r in rows.items():
        f = lambda v: round(float(np.mean([x for x in v if x is not None])), 3) if any(x is not None for x in v) else None
        summ[n] = {"episodes": r["n"], "detected": r["detected"], "xy_err_cm_mean": f(r["err_cm"]),
                   "xy_err_cm_max": round(max(r["err_cm"]), 1), "sam_iou_vs_gt": f(r["sam_iou"]),
                   "render_iou_vs_gt": f(r["render_iou"])}
    allerr = [x for r in rows.values() for x in r["err_cm"]]
    summ["_all"] = {"xy_err_cm_mean": round(float(np.mean(allerr)), 2),
                    "frac_within_3cm": round(float(np.mean(np.array(allerr) < 3)), 3),
                    "render_iou_mean": round(float(np.mean([x for r in rows.values() for x in r["render_iou"]
                                                            if x is not None])), 3)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summ, open(a.out, "w"), indent=1)
    for n, v in summ.items():
        print(n, v)


if __name__ == "__main__":
    main()
