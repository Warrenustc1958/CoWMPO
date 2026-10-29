"""S5 shape identification (Builder side): fit declared size parameters to first-frame evidence masks.

Objective for a parameter of object k = mean over FIT episodes of IoU(world render of k after S5 pose
estimation, SAM2 mask of k). Coordinate search with shrinking multiplicative steps; every candidate is
compiled and pose-estimated in its own subprocess (one GL context per process). Only Builder-visible
inputs are used (program, evidence masks); target GT is never read here.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cowm.world.compile_world import compile_world

EVIDENCE = "/data_cf/cowmpo/evidence/libero_goal_dev3"
STEPS = [(0.4, 0.6, 1.6, 2.5), (0.6, 0.8, 1.25, 1.6), (0.85, 0.93, 1.08, 1.18), (0.95, 1.05)]
MIN_GAIN = 0.003            # a size change must buy at least this much IoU (ties keep the current value)
MIN_GAIN_TRUNCATED = 0.015  # objects cut by the image border: their out-of-frame extent is barely observable


def truncated_objects(masks, eps, frac=0.5):
    """Objects whose evidence mask touches the image border in more than `frac` of the fit episodes."""
    hits = {}
    for ep in eps:
        m = np.load(os.path.join(masks, ep, "masks.npz"))
        for n in m.files:
            mk = m[n]
            border = mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any()
            hits.setdefault(n, []).append(bool(border))
    return {n for n, v in hits.items() if np.mean(v) > frac}


def obj_of(param, objects):
    """Object a size parameter belongs to: the object whose attrs reference it (from world.json params)."""
    for o in objects:
        if param in o.get("param_refs", []):
            return o["name"]
    return None


def evaluate(program, values, eps, masks, work, contact_rule="legacy"):
    wdir, sdir = os.path.join(work, "world"), os.path.join(work, "state")
    compile_world(program, wdir, values, contact_rule=contact_rule)
    subprocess.run([sys.executable, "-m", "cowm.init.estimate_state", "--world", wdir, "--evidence", EVIDENCE,
                    "--masks", masks, "--out", sdir, "--episodes", ",".join(eps)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    per = {}
    for ep in eps:
        st = json.load(open(os.path.join(sdir, ep, "state.json")))["objects"]
        for n, v in st.items():
            if v.get("iou") is not None:
                per.setdefault(n, []).append(v["iou"])
    return {n: float(np.mean(v)) for n, v in per.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--program", required=True)
    p.add_argument("--masks", required=True, help="EvidenceAdapter output (<episode>/masks.npz)")
    p.add_argument("--out", required=True)
    p.add_argument("--fit", required=True, help="comma list of FIT episodes")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--init", default="", help="previous shape_fit.json to warm start from")
    p.add_argument("--contact-rule", default="legacy", choices=["legacy", "object_table_pair"],
                   help="kept for every candidate and for world_fit (recorded in shape_fit.json)")
    a = p.parse_args()
    eps = a.fit.split(",")
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "base_world")
    compile_world(a.program, base, contact_rule=a.contact_rule)
    wj = json.load(open(os.path.join(base, "world.json")))
    params = wj["params"]
    # size parameters = declared params referenced by an object's geometry (dynamics params excluded)
    shape = {pn: obj_of(pn, wj["objects"]) for pn in params
             if obj_of(pn, wj["objects"]) and "material_class" not in params[pn] and "damping" not in pn}
    cur = {pn: params[pn]["value"] for pn in shape}
    if a.init:   # warm start from a previous fit (values clipped to the current bounds)
        prev = json.load(open(a.init))["values"]
        for pn, v in prev.items():
            if pn in cur:
                lo, hi = params[pn]["bounds"]
                cur[pn] = min(max(float(v), lo), hi)
    trunc = truncated_objects(a.masks, eps)
    print("SHAPE_PARAMS", shape, "border-truncated:", sorted(trunc), flush=True)
    log = []
    n_job = [0]

    def job(values):
        n_job[0] += 1
        work = os.path.join(a.out, "jobs", f"j{n_job[0]:04d}")
        try:
            return evaluate(a.program, values, eps, a.masks, work, a.contact_rule)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    best_score = job(dict(cur))
    print("BASE", {k: round(v, 3) for k, v in best_score.items()}, flush=True)
    for rnd, factors in enumerate(STEPS):
        cands = []
        for pn in shape:
            lo, hi = params[pn]["bounds"]
            for f in factors:
                v = min(max(cur[pn] * f, lo), hi)
                if abs(v - cur[pn]) > 1e-6:
                    cands.append((pn, v))
        with ThreadPoolExecutor(a.jobs) as ex:
            res = list(ex.map(lambda c: job({**cur, c[0]: c[1]}), cands))
        new = dict(cur)
        for pn in shape:
            obj = shape[pn]
            gain = MIN_GAIN_TRUNCATED if obj in trunc else MIN_GAIN
            best_v, best_s = cur[pn], best_score.get(obj, 0.0)
            for (q, v), r in zip(cands, res):
                if q == pn and r.get(obj, 0.0) > best_s + gain:
                    best_v, best_s = v, r[obj]
            new[pn] = best_v
        # combined update re-evaluated (params of one object interact); per object, keep it only if that
        # object's score did not drop, otherwise revert the object's params to the previous round
        trial = job(dict(new))
        for obj in set(shape.values()):
            if trial.get(obj, 0.0) + 1e-9 < best_score.get(obj, 0.0):
                for pn, o in shape.items():
                    if o == obj:
                        new[pn] = cur[pn]
        if new != cur:
            cur = new
            best_score = job(dict(cur))
        else:
            best_score = trial if trial and all(trial.get(o, 0) >= best_score.get(o, 0) for o in best_score) \
                else best_score
        log.append({"round": rnd, "values": dict(cur), "score": best_score})
        print("ROUND", rnd, {k: round(v, 3) for k, v in cur.items()},
              {k: round(v, 3) for k, v in best_score.items()}, flush=True)
    json.dump({"fit_episodes": eps, "shape_params": shape, "values": cur, "score": best_score, "log": log,
               "contact_rule": a.contact_rule}, open(os.path.join(a.out, "shape_fit.json"), "w"), indent=1)
    compile_world(a.program, os.path.join(a.out, "world_fit"), cur, contact_rule=a.contact_rule)
    print("SHAPE_DONE", os.path.join(a.out, "world_fit"))


if __name__ == "__main__":
    main()
