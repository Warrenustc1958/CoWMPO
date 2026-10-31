# Copyright 2026 The RLinf Authors.
"""Transition-level selection for v35 far-end candidates (Builder side; public RGB + Builder models only, no GT).
Runs the deceleration-onset diagnostic (diag_rack_contact_rgb_v31 rule) on all 33 public probes (27 old + 6 far34)
for each v35 candidate, then applies the rule fixed in rack_far_dev_v35/freeze.json.
usage: python score_rack_far_v35.py
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

C = Path(__file__).resolve().parent
R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
FAM = R / "rack_far_dev_v36"
DIAG = R / "diag_rack_all_v36"
OUT = R / "rack_far_score_v36"
RUNNER = '''import sys
import diag_rack_contact_rgb_v31 as D
D.SETS = dict(D.SETS, far34=("rack_far_dense_plan_v34", "codex_rack_far_dense_v34"))
D.OUT = D.R / "diag_rack_all_v36"
D.run(sys.argv[1])
'''


def name(c):
    return f"{c.parent.parent.name}__{c.parent.name}"


def main():
    OUT.mkdir(exist_ok=True)
    (C / "_diag_all_v36.py").write_text(RUNNER)
    cands = sorted(FAM.glob("cand??/result.json"))

    def run(c):
        if (DIAG / f"{name(c)}.json").exists():
            return
        with (OUT / f"{name(c)}.log").open("w") as log:
            r = subprocess.run([sys.executable, str(C / "_diag_all_v36.py"), str(c)], stdout=log,
                               stderr=subprocess.STDOUT, cwd=str(C))
        if r.returncode:
            raise RuntimeError((str(c), r.returncode))
    with ThreadPoolExecutor(13) as pool:
        list(pool.map(run, cands))
    fz = json.loads((FAM / "freeze.json").read_text())
    rows = []
    for c in cands:
        d = json.loads((DIAG / f"{name(c)}.json").read_text())["probes"]
        res = json.loads(c.read_text())
        err = {k: abs(v["sim_dec0.5"] - v["obs_dec0.5"]) for k, v in d.items()
               if v["obs_dec0.5"] is not None and v["obs_dec0.5"] <= 240 and v["sim_dec0.5"] is not None}
        miss = sorted(k for k, v in d.items() if v["sim_dec0.5"] is None and v["obs_dec0.5"] is not None)
        old = [e for k, e in err.items() if not k.startswith("far34")]
        far = {k: e for k, e in err.items() if k.startswith("far34")}
        rows.append({"cand": name(c), "far_ext": res["far_ext"], "far_dz": res["far_dz"],
                     "trans_err_all": round(float(np.mean(list(err.values()))), 3), "n_valid": len(err),
                     "trans_err_old27": round(float(np.mean(old)), 3), "err_max": max(err.values()),
                     "far_err": far, "far_sim_first": {k: v["sim_first"] for k, v in d.items() if k.startswith("far34")},
                     "public_iou_all": round(res["public_iou_all"], 4), "missing_sim": miss})
    ref = next(r for r in rows if r["cand"].endswith("cand00"))
    g = fz["guardrails"]
    for r in rows:
        r["guard_ok"] = (r["trans_err_old27"] <= ref["trans_err_old27"] + g["old27_trans_err_rise"] and
                         r["public_iou_all"] >= ref["public_iou_all"] - g["public_static_iou_drop"])
    rows.sort(key=lambda r: (not r["guard_ok"], r["trans_err_all"]))
    (OUT / "score.json").write_text(json.dumps({"rule": fz["selection"], "guardrails": g, "reference": ref["cand"],
                                                "candidates": rows, "GT_inputs": False}, indent=1))
    for r in rows:
        print(json.dumps({k: r[k] for k in ("cand", "far_ext", "far_dz", "trans_err_all", "trans_err_old27",
                                             "err_max", "far_err", "public_iou_all", "guard_ok")}), flush=True)
    print("FAR_SCORE_V36_DONE", flush=True)


if __name__ == "__main__":
    main()
