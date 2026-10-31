# Copyright 2026 The RLinf Authors.
"""Builder diagnostic (public RGB tracks only, no GT, no Target numbers): why the two acc26 y-060 probes have an
observed stall at 231/240 while other probes stall at 211-221. Slide-off-edge vs stall-criterion failure.
Per probe, per window step: observed robot-mask centroid (row/col) and area inside the rack ROI, tracker union/added
areas, and the same centroid for the v25c01 model render (cached in rack_dense_score_v27/sim).
usage: python diag_rack_ym060_v28.py
"""
import json
from pathlib import Path

import numpy as np

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
PLAN = R / "rack_accept_dense_plan_v27"
TR = R / "rack_accept_dense_v27_tracks/frames"
SIM = R / "rack_dense_score_v27/sim/rack_global_dev_v27__cand01.npz"
SCORE = R / "rack_dense_score_v27/score.json"
OUT = R / "diag_rack_ym060_v28.json"
EP = "libero_goal_t0_i0"
KEYS = ["acc26_x+070_y-060", "acc26_x+090_y-060", "acc26_x+020_y-040", "acc26_x+110_y-040", "acc26_x+110_y+040"]


def stats(ms):
    ys, xs = np.mgrid[:ms.shape[1], :ms.shape[2]]
    a = ms.sum((1, 2))
    return (ms * ys).sum((1, 2)) / np.maximum(a, 1), (ms * xs).sum((1, 2)) / np.maximum(a, 1), a


def main():
    y0, y1, x0, x1 = json.loads(SCORE.read_text())["roi"]
    sim = np.load(SIM)
    rep = {"roi": [y0, y1, x0, x1], "GT_inputs": False, "Target_numbers_read": False, "probes": {}}
    for key in KEYS:
        p = json.loads((PLAN / key / "probe.json").read_text())
        a, b = p["dense_window"]
        steps = list(range(a, b + 1))
        obs, meta = [], []
        for t in steps:
            fr = TR / key / f"{EP}__f{t:03d}"
            with np.load(fr / "masks.npz") as z:
                obs.append(z["robot"][y0:y1, x0:x1])
            meta.append(json.loads((fr / "association.json").read_text())["objects"]["robot"])
        obs = np.stack(obs)
        cy, cx, ar = stats(obs)
        scy, scx, sar = stats(sim[key])
        # full-frame robot mask lowest row (gripper tip proxy) for the observed track
        rows = []
        for i, t in enumerate(steps):
            rows.append({"step": t, "obs_cy": round(float(cy[i]), 2), "obs_cx": round(float(cx[i]), 2),
                         "obs_area": int(ar[i]), "union": meta[i]["union_area"], "added": meta[i]["added_area"],
                         "sim_cy": round(float(scy[i]), 2), "sim_cx": round(float(scx[i]), 2), "sim_area": int(sar[i])})
        rep["probes"][key] = rows
        print("==", key, "window", a, b)
        for r in rows[::3]:
            print("  %3d obs cy %6.2f cx %6.2f area %4d (union %5d added %4d) | sim cy %6.2f cx %6.2f area %4d" % (
                r["step"], r["obs_cy"], r["obs_cx"], r["obs_area"], r["union"], r["added"], r["sim_cy"], r["sim_cx"],
                r["sim_area"]), flush=True)
    OUT.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
