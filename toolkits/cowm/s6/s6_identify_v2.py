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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cowm.s6.cmaes import cmaes
from cowm import roles as R
from cowm.s6.g2_replay import G2_DYN_PX, summarize, verify_manifest, require_reference_bindings


INVALID_J = 1e9   # candidate violating a completeness check: never selected


def objective(s):
    """Fit loss over candidates that satisfy the G2 completeness checks (every episode measured, no missing
    world object, lost-frame rate). Incomplete candidates get INVALID_J, so the optimizer cannot prefer a world
    that drops hard evidence; if every candidate is invalid the run reports identification failure."""
    c = s.get("checks", {})
    required = ("every_episode_complete", "no_missing_world_objects", "lost_frames", "evidence_coverage",
                "initial_states_valid", "finite_metrics")
    if not all(c.get(k) for k in required):
        return INVALID_J
    if not all(s.get(k) is not None and np.isfinite(s[k]) for k in ("dyn_err_px", "static_iou")):
        return INVALID_J
    return s["dyn_err_px"] / G2_DYN_PX + 2.0 * (1.0 - s["static_iou"])


def require_episode_set(results, expected):
    """Reject omitted, duplicate or substituted subprocess episode records before aggregation."""
    actual = [r.get("episode") for r in results]
    if len(expected) != len(set(expected)) or Counter(actual) != Counter(expected):
        raise ValueError(f"G2 episode set mismatch: expected {expected}, received {actual}")


def g2_eval(pool, world, state, probes, masks, eps, params, tmp, world_tree_sha256, validity_path, chunks=4):
    parts = [eps[i::chunks] for i in range(chunks) if eps[i::chunks]]

    def run(part):
        fd, out = tempfile.mkstemp(suffix=".json", dir=tmp)
        os.close(fd)
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "g2_replay.py"), "--world", world, "--state", state, "--probes", probes,
               "--masks", masks, "--out", out, "--episodes", ",".join(part),
               "--world-tree-sha256", world_tree_sha256, "--require-bound-reference", "--validity", validity_path]
        if params:
            cmd += ["--params", json.dumps(params)]
        try:
            completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if completed.returncode:
                raise RuntimeError(f"G2 replay failed ({completed.returncode}): {completed.stderr[-8000:]}")
            r = json.load(open(out))["episodes"]
            require_episode_set(r, part)
        finally:
            os.remove(out)
        return r

    res = [r for rs in pool.map(run, parts) for r in rs]
    require_episode_set(res, eps)
    s = summarize(res)   # shared G2 verdict (static, dynamic, terminal, coverage; audit 10.28)
    return s, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-world", required=True)
    p.add_argument("--ref-state", required=True)
    p.add_argument("--shape-world", required=True)
    p.add_argument("--shape-state", required=True)
    p.add_argument("--ref-tree-sha256", required=True, help="directory hash of the frozen reference world")
    p.add_argument("--shape-tree-sha256", required=True, help="directory hash of the frozen shape world")
    p.add_argument("--probes", required=True)
    p.add_argument("--probe-masks", required=True)
    p.add_argument("--validity", required=True, help="new frozen manifest bound to reference assets and states")
    p.add_argument("--ident", required=True, help="identifiability.json")
    p.add_argument("--out", required=True)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--popsize", type=int, default=6)
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--allow-dev-fit", action="store_true",
                   help="debug only: fit on exposed dev trials when no fit-role probes exist (report is labelled)")
    a = p.parse_args()
    from cowm.world.artifact import verify_tree
    verify_tree(a.ref_world, a.ref_tree_sha256)
    verify_tree(a.shape_world, a.shape_tree_sha256)
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
    manifest = json.load(open(a.validity))
    require_reference_bindings(manifest, fit + held)
    verify_manifest(manifest, a.probes, a.probe_masks, fit + held)
    from cowm.init.observation import state_issues
    invalid_states = {}
    for label, world, state_root, world_hash in (
            ("reference", a.ref_world, a.ref_state, a.ref_tree_sha256),
            ("shape", a.shape_world, a.shape_state, a.shape_tree_sha256)):
        world_doc = json.load(open(os.path.join(world, "world.json")))
        names_ = [o["name"] for o in world_doc["objects"]]
        for ep in fit + held:
            state_doc = json.load(open(os.path.join(state_root, ep, "state.json")))
            issues = state_issues(state_doc["objects"], names_)
            if state_doc.get("episode") != ep:
                issues["_episode"] = ["state_episode_mismatch"]
            if (state_doc.get("world_tree_sha256") != world_hash or
                    state_doc.get("world_sha256") != world_doc["world_sha256"]):
                issues["_world"] = ["state_world_or_assets_mismatch"]
            if issues:
                invalid_states.setdefault(label, {})[ep] = issues
    if invalid_states:
        json.dump({"identification_failed": True, "reason": "invalid_initial_states",
                   "fit_episodes": fit, "dev_episodes": held, "invalid_initial_states": invalid_states},
                  open(os.path.join(a.out, "s6_preflight.json"), "w"), indent=1)
        raise SystemExit("S6 preflight failed; initial-state uncertainty cannot be fixed by dynamics search")
    ident = json.load(open(a.ident))
    names = [n for n, v in ident.items() if v["identifiable"]]
    spec = json.load(open(os.path.join(a.shape_world, "world.json")))["params"]
    bounds = {n: spec[n]["bounds"] for n in names}
    print("FIT_PARAMS", names, "not identifiable (kept at prior):", [n for n in ident if n not in names], flush=True)
    pool = ThreadPoolExecutor(a.jobs)
    out, log = {}, {}

    def to_vals(z):
        return {n: float(bounds[n][0] + zi * (bounds[n][1] - bounds[n][0])) for n, zi in zip(names, z)}

    variants = [("default", a.ref_world, a.ref_state, False, a.ref_tree_sha256),
                ("param", a.ref_world, a.ref_state, True, a.ref_tree_sha256),
                ("param+struct", a.shape_world, a.shape_state, True, a.shape_tree_sha256)]
    for name, world, state, do_fit, world_hash in variants:
        params = {}
        if do_fit and names:
            x0 = [(spec[n]["init"] - bounds[n][0]) / (bounds[n][1] - bounds[n][0]) for n in names]

            def batch(Z):
                with ThreadPoolExecutor(len(Z)) as ex:
                    ss = list(ex.map(lambda z: g2_eval(pool, world, state, a.probes, a.probe_masks, fit,
                                                         to_vals(z), tmp, world_hash, a.validity)[0], Z))
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
        fs, fres = g2_eval(pool, world, state, a.probes, a.probe_masks, fit, params, tmp, world_hash, a.validity)
        hs, hres = g2_eval(pool, world, state, a.probes, a.probe_masks, held, params, tmp, world_hash, a.validity)
        with open(os.path.join(a.out, name.replace("+", "_") + "_episodes.json"), "w") as stream:
            json.dump({"params": params, "world_tree_sha256": world_hash,
                       "fit": fres, "dev": hres}, stream, indent=1)
        out[name] = {"params": params, "world": world, "state": state, "fit": fs, "fit_J": objective(fs),
                     "heldout": hs, "heldout_J": objective(hs)}
        print("VARIANT", name, "fit J", round(objective(fs), 3), "heldout",
              {k: round(v, 3) if isinstance(v, float) else v for k, v in hs.items()}, "J", round(objective(hs), 3),
              flush=True)
    json.dump({"fit_episodes": fit, "heldout_episodes": held, "debug_dev_only": debug_dev_only,
               "validity": a.validity, "ref_tree_sha256": a.ref_tree_sha256,
               "shape_tree_sha256": a.shape_tree_sha256,
               "splits": R.load()["path"], "fitted_params": names, "identifiability": ident,
               "variants": out, "cmaes_log": log}, open(os.path.join(a.out, "s6_report.json"), "w"), indent=1)
    print("S6V2_DONE")


if __name__ == "__main__":
    main()
