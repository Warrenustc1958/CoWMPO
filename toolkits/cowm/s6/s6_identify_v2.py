"""S6 identification v2 (Builder side): default template vs param-fit vs param+struct-fit, CMA-ES.

  default      : reference-template world, declared material priors
  param        : reference-template world + CMA-ES over the identifiable dynamics parameters
  param+struct : S5 shape-identified world + CMA-ES over the identifiable dynamics parameters
Objective on FIT episodes only: J = dyn_err_px / G2_DYN_PX + 2 * (1 - static_iou).
Which parameters are fitted comes from the identifiability check (simulation only); the others keep priors.
Every evaluation runs as g2_replay subprocesses over episode chunks (one GL context per process).
The RM / success / verifier / G2 thresholds are fixed; identification never touches them.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cowm.s6.cmaes import cmaes
from cowm.s6.g2_replay import G2_DYN_PX, G2_STATIC_IOU


def objective(s):
    return s["dyn_err_px"] / G2_DYN_PX + 2.0 * (1.0 - s["static_iou"])


def g2_eval(pool, world, state, probes, masks, eps, params, tmp, chunks=4):
    parts = [eps[i::chunks] for i in range(chunks) if eps[i::chunks]]

    def run(part):
        fd, out = tempfile.mkstemp(suffix=".json", dir=tmp)
        os.close(fd)
        cmd = [sys.executable, "-m", "cowm.s6.g2_replay", "--world", world, "--state", state, "--probes", probes,
               "--masks", masks, "--out", out, "--episodes", ",".join(part)]
        if params:
            cmd += ["--params", json.dumps(params)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        r = json.load(open(out))["episodes"]
        os.remove(out)
        return r

    res = [r for rs in pool.map(run, parts) for r in rs]
    f = lambda k: float(np.mean([r[k] for r in res if r[k] is not None]))
    s = {"n": len(res), "static_iou": f("static_iou"), "dyn_err_px": f("dyn_err_px"),
         "final_obj_iou": f("final_obj_iou")}
    s["G2_target_evidence"] = bool(s["static_iou"] >= G2_STATIC_IOU and s["dyn_err_px"] <= G2_DYN_PX)
    return s, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-world", required=True)
    p.add_argument("--ref-state", required=True)
    p.add_argument("--shape-world", required=True)
    p.add_argument("--shape-state", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--probe-masks", required=True)
    p.add_argument("--ident", required=True, help="identifiability.json")
    p.add_argument("--out", required=True)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--popsize", type=int, default=6)
    p.add_argument("--jobs", type=int, default=24)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "tmp")
    os.makedirs(tmp, exist_ok=True)
    eps = sorted(os.listdir(a.probes))
    fit = [e for e in eps if not e.endswith("_i2")]
    held = [e for e in eps if e.endswith("_i2")]
    ident = json.load(open(a.ident))
    names = [n for n, v in ident.items() if v["identifiable"]]
    spec = json.load(open(os.path.join(a.shape_world, "world.json")))["params"]
    bounds = {n: spec[n]["bounds"] for n in names}
    print("FIT_PARAMS", names, "not identifiable (kept at prior):", [n for n in ident if n not in names], flush=True)
    pool = ThreadPoolExecutor(a.jobs)
    out, log = {}, {}

    def to_vals(z):
        return {n: float(bounds[n][0] + zi * (bounds[n][1] - bounds[n][0])) for n, zi in zip(names, z)}

    variants = [("default", a.ref_world, a.ref_state, False), ("param", a.ref_world, a.ref_state, True),
                ("param+struct", a.shape_world, a.shape_state, True)]
    for name, world, state, do_fit in variants:
        params = {}
        if do_fit and names:
            x0 = [(spec[n]["init"] - bounds[n][0]) / (bounds[n][1] - bounds[n][0]) for n in names]

            def batch(Z):
                with ThreadPoolExecutor(len(Z)) as ex:
                    ss = list(ex.map(lambda z: g2_eval(pool, world, state, a.probes, a.probe_masks, fit,
                                                         to_vals(z), tmp)[0], Z))
                return [objective(s) for s in ss]

            z, fbest, glog = cmaes(None, x0, sigma0=0.3, popsize=a.popsize, generations=a.generations,
                                   f_batch=batch)
            params = to_vals(z)
            log[name] = glog
            print("CMAES", name, {k: round(v, 3) for k, v in params.items()}, "fit J", round(fbest, 3), flush=True)
        fs, _ = g2_eval(pool, world, state, a.probes, a.probe_masks, fit, params, tmp)
        hs, hres = g2_eval(pool, world, state, a.probes, a.probe_masks, held, params, tmp)
        out[name] = {"params": params, "world": world, "state": state, "fit": fs, "fit_J": objective(fs),
                     "heldout": hs, "heldout_J": objective(hs)}
        print("VARIANT", name, "fit J", round(objective(fs), 3), "heldout",
              {k: round(v, 3) if isinstance(v, float) else v for k, v in hs.items()}, "J", round(objective(hs), 3),
              flush=True)
    json.dump({"fit_episodes": fit, "heldout_episodes": held, "fitted_params": names, "identifiability": ident,
               "variants": out, "cmaes_log": log}, open(os.path.join(a.out, "s6_report.json"), "w"), indent=1)
    print("S6V2_DONE")


if __name__ == "__main__":
    main()
