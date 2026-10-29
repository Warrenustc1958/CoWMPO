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
from cowm import roles as R
from cowm.s6.g2_replay import G2_DYN_PX, summarize


INVALID_J = 1e9   # candidate violating a completeness check: never selected


def objective(s):
    """Fit loss over candidates that satisfy the G2 completeness checks (every episode measured, no missing
    world object, lost-frame rate). Incomplete candidates get INVALID_J, so the optimizer cannot prefer a world
    that drops hard evidence; if every candidate is invalid the run reports identification failure."""
    c = s.get("checks", {})
    if not (c.get("every_episode_complete") and c.get("no_missing_world_objects") and c.get("lost_frames")):
        return INVALID_J
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
    s = summarize(res)   # shared G2 verdict (static, dynamic, terminal, coverage; audit 10.28)
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
    p.add_argument("--allow-dev-fit", action="store_true",
                   help="debug only: fit on exposed dev trials when no fit-role probes exist (report is labelled)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "tmp")
    os.makedirs(tmp, exist_ok=True)
    eps = sorted(os.listdir(a.probes))
    # data roles from the profile split (never from file names); gate/test episodes are refused outright
    R.require(eps, {"fit", "dev"}, "s6_identify_v2 probe set")
    fit, held = R.select(eps, "fit"), R.select(eps, "dev")
    debug_dev_only = False
    if not fit:
        if not a.allow_dev_fit:
            raise SystemExit("no fit-role episodes in the probe set (exposed episodes are dev forever); collect "
                             "probes on unexposed fit trials, or pass --allow-dev-fit for a labelled debug run")
        # debug only: fit on dev trials 0-1, evaluate on dev trial 2; the report is marked debug_dev_only
        fit = [e for e in held if R.parse(e)[1] != 2]
        held = [e for e in held if R.parse(e)[1] == 2]
        debug_dev_only = True
    print("ROLES fit", len(fit), "held(dev)", len(held), "debug_dev_only", debug_dev_only, flush=True)
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
            if fbest >= INVALID_J:
                print("CMAES", name, "IDENTIFICATION_FAILED: no candidate satisfied the completeness checks",
                      flush=True)
                out[name] = {"identification_failed": True, "world": world, "state": state}
                continue
            print("CMAES", name, {k: round(v, 3) for k, v in params.items()}, "fit J", round(fbest, 3), flush=True)
        fs, _ = g2_eval(pool, world, state, a.probes, a.probe_masks, fit, params, tmp)
        hs, hres = g2_eval(pool, world, state, a.probes, a.probe_masks, held, params, tmp)
        out[name] = {"params": params, "world": world, "state": state, "fit": fs, "fit_J": objective(fs),
                     "heldout": hs, "heldout_J": objective(hs)}
        print("VARIANT", name, "fit J", round(objective(fs), 3), "heldout",
              {k: round(v, 3) if isinstance(v, float) else v for k, v in hs.items()}, "J", round(objective(hs), 3),
              flush=True)
    json.dump({"fit_episodes": fit, "heldout_episodes": held, "debug_dev_only": debug_dev_only,
               "splits": R.load()["path"], "fitted_params": names, "identifiability": ident,
               "variants": out, "cmaes_log": log}, open(os.path.join(a.out, "s6_report.json"), "w"), indent=1)
    print("S6V2_DONE")


if __name__ == "__main__":
    main()
