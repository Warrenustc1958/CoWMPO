# Copyright 2026 The RLinf Authors.
"""Transition-level selection for v31 rack candidates (Builder side; public RGB + Builder models only, no GT, no
Target numbers). Lesson from code-world-model repair work (WorldCoder / GIF-MCTS style): fit the transition event that
is judged, not an appearance proxy. Acceptance judges first contact; ROI mask IoU saturates near the -y edge.
Per candidate and public probe: dec = first step after the descent peak where frame-to-frame change < 0.5 x peak,
same rule on public RGB (obs) and on the candidate's rendered robot mask (sim) (diag_rack_contact_rgb_v31.py).
  trans_err  mean |sim_dec - obs_dec| over valid probes; a probe is excluded iff obs_dec is None or > 240 (no blocked
             moment, the arm keeps sliding) - rule fixed in freeze.json before any candidate is scored
Selection: lowest trans_err among candidates passing all guardrails vs the identity candidate (v29 cand10):
  static IoU >= ref - .01;  v12 front-row robot IoU >= ref - .005;  dense ROI-IoU loss (score_rack_dense_v31) <=
  ref + .02.  Ties -> lower dense loss.
usage: python score_rack_transition_v31.py
"""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
C = Path(__file__).resolve().parent
FAM = R / "rack_rib_dev_v31"
DIAG = R / "diag_rack_contact_rgb_v31"
DENSE = R / "rack_dense_score_v31/score.json"
OUT = R / "rack_transition_score_v31"
RULE = {"signal": "dec0.5", "exclude": "obs_dec is None or obs_dec > 240",
        "guard": {"static_drop": .01, "front_drop": .005, "dense_loss_rise": .02}}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name(c):
    return f"{c.parent.parent.name}__{c.parent.name}"


def main():
    OUT.mkdir(exist_ok=True)
    cands = sorted(FAM.glob("cand??/result.json"))
    (OUT / "freeze.json").write_text(json.dumps({"rule": RULE, "candidates": [str(c) for c in cands],
                                                 "script_sha256": sha(__file__),
                                                 "diag_sha256": sha(C / "diag_rack_contact_rgb_v31.py"),
                                                 "GT_inputs": False, "Target_numbers_read": False}, indent=1))

    def run(c):
        if (DIAG / f"{name(c)}.json").exists():
            return
        with (OUT / f"{name(c)}.log").open("w") as log:
            r = subprocess.run([sys.executable, str(C / "diag_rack_contact_rgb_v31.py"), str(c)],
                               stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((str(c), r.returncode))
    with ThreadPoolExecutor(49) as pool:
        list(pool.map(run, cands))
    dense = {r["cand"]: r for r in json.loads(DENSE.read_text())["candidates"]}
    front = {}
    for c in cands:
        res = json.loads(c.read_text())
        front[name(c)] = float(np.mean([q["mean_iou"] for q in res["responses"]]))
    rows = []
    for c in cands:
        d = json.loads((DIAG / f"{name(c)}.json").read_text())["probes"]
        per = {k: (v["sim_dec0.5"], v["obs_dec0.5"]) for k, v in d.items()}
        valid = {k: abs(s - o) for k, (s, o) in per.items() if o is not None and o <= 240 and s is not None}
        res = json.loads(c.read_text())
        rows.append({"cand": name(c), "params": {k: res[k] for k in ("rib_cx", "rib_hw", "rib_dz", "rib_ye",
                                                                       "halftop_xb")},
                     "trans_err": round(float(np.mean(list(valid.values()))), 3), "n_valid": len(valid),
                     "excluded": sorted(set(per) - set(valid)),
                     "err_neg_y": {k: valid[k] for k in valid if "_y-" in k},
                     "err_max": max(valid.values()), "static_iou": round(res["static_iou"], 4),
                     "front_robot_iou": round(front[name(c)], 4), "dense_loss": dense[name(c)]["loss"]})
    ref = next(r for r in rows if r["cand"].endswith("cand00"))
    g = RULE["guard"]
    for r in rows:
        r["guard_ok"] = (r["static_iou"] >= ref["static_iou"] - g["static_drop"] and
                         r["front_robot_iou"] >= ref["front_robot_iou"] - g["front_drop"] and
                         r["dense_loss"] <= ref["dense_loss"] + g["dense_loss_rise"])
    rows.sort(key=lambda r: (not r["guard_ok"], r["trans_err"], r["dense_loss"]))
    (OUT / "score.json").write_text(json.dumps({"rule": RULE, "reference": ref["cand"], "candidates": rows,
                                                "GT_inputs": False, "Target_numbers_read": False}, indent=1))
    for r in rows[:12] + [ref]:
        print(json.dumps({k: r[k] for k in ("cand", "params", "trans_err", "err_max", "err_neg_y", "static_iou",
                                             "front_robot_iou", "dense_loss", "guard_ok")}), flush=True)
    print("TRANS_V31_DONE", flush=True)


if __name__ == "__main__":
    main()
