"""S6 identification (Builder side, plan §9.7-D): default template vs param-fit vs param+struct-fit.

Evidence: S5 first-frame masks + probe frames the Target returned (RGB only).
  default      : reference world, declared material priors
  param        : grid over <ceramic>.friction (runtime override), state from S5
  param+struct : grid over bowl_radius (recompiled world + re-estimated S5 state) x friction
Selection objective on FIT episodes only (trials i0, i1):  J = dyn_err_px / G2_DYN_PX + 2 * (1 - static_iou)
Reported on HELD-OUT episodes (trial i2) with the G2 gate. The RM / success / verifier / G2 thresholds are
fixed and never touched by identification.
Every world evaluation runs in a fresh subprocess (one offscreen GL context per process): Scenes created
late in a long-lived process rendered wrong segmentation after in-process recompiles.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from cowm.s6.g2_replay import G2_DYN_PX
from cowm.world.compile_world import compile_world

EVIDENCE = "/data_cf/cowmpo/evidence/libero_goal_dev3"


def objective(s):
    return s["dyn_err_px"] / G2_DYN_PX + 2.0 * (1.0 - s["static_iou"])


def g2(a, world, state, eps, mu, out):
    cmd = [sys.executable, "-m", "cowm.s6.g2_replay", "--world", world, "--state", state, "--probes", a.probes,
           "--masks", a.probe_masks, "--out", out, "--episodes", ",".join(eps)]
    if mu is not None:
        cmd += ["--friction", json.dumps({a.material: mu})]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out))["summary"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--program", required=True, help="reference world program")
    p.add_argument("--ref-world", required=True)
    p.add_argument("--s5-masks", required=True)
    p.add_argument("--s5-state", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--probe-masks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--material", default="ceramic")
    p.add_argument("--frictions", default="0.4,0.7,1.0")
    p.add_argument("--radii", default="0.05,0.06,0.07,0.08")
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eps = sorted(os.listdir(a.probes))
    fit = [e for e in eps if not e.endswith("_i2")]
    held = [e for e in eps if e.endswith("_i2")]
    fr = [float(x) for x in a.frictions.split(",")]
    radii = [float(x) for x in a.radii.split(",")]
    key = f"{a.material}.friction"

    # 1) structural candidates: compile + re-estimate S5 state (one subprocess each)
    worlds = {}
    for r in radii:
        tag = f"r{int(round(r * 1000)):03d}"
        wdir, sdir = os.path.join(a.out, f"world_{tag}"), os.path.join(a.out, f"state_{tag}")
        compile_world(a.program, wdir, {"bowl_radius": r})
        worlds[r] = (wdir, sdir)

    def est(r):
        wdir, sdir = worlds[r]
        subprocess.run([sys.executable, "-m", "cowm.init.estimate_state", "--world", wdir, "--evidence", EVIDENCE,
                        "--masks", a.s5_masks, "--out", sdir, "--episodes", ",".join(eps)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(est, radii))

    # 2) fit-set grid (all candidates in parallel)
    jobs = [("default", a.ref_world, a.s5_state, None, None)]
    jobs += [("param", a.ref_world, a.s5_state, mu, None) for mu in fr]
    jobs += [("param+struct", worlds[r][0], worlds[r][1], mu, r) for r in radii for mu in fr]

    def run(j):
        name, w, s, mu, r = j
        out = os.path.join(a.out, f"g2_fit_{name}_r{r}_mu{mu}.json")
        sm = g2(a, w, s, fit, mu, out)
        return {"variant": name, "world": w, "state": s, "friction": mu, "radius": r, "fit": sm, "J": objective(sm)}
    with ThreadPoolExecutor(a.jobs) as ex:
        log = list(ex.map(run, jobs))
    for rec in log:
        print("FIT", rec["variant"], "r", rec["radius"], "mu", rec["friction"],
              {k: round(v, 3) if isinstance(v, float) else v for k, v in rec["fit"].items()},
              "J", round(rec["J"], 3), flush=True)

    # 3) select per variant on FIT, report on HELD-OUT
    best = {v: min((r for r in log if r["variant"] == v), key=lambda r: r["J"])
            for v in ("default", "param", "param+struct")}
    out = {}
    for v, b in best.items():
        hs = g2(a, b["world"], b["state"], held, b["friction"], os.path.join(a.out, f"g2_heldout_{v}.json"))
        params = {}
        if b["friction"] is not None:
            params[key] = b["friction"]
        if b["radius"] is not None:
            params["bowl_radius"] = b["radius"]
        out[v] = {"params": params, "world": b["world"], "state": b["state"], "fit_J": b["J"],
                  "heldout": hs, "heldout_J": objective(hs)}
        print("HELDOUT", v, params, {k: round(x, 3) if isinstance(x, float) else x for k, x in hs.items()},
              "J", round(objective(hs), 3), flush=True)
    json.dump({"fit_episodes": fit, "heldout_episodes": held, "search": log, "variants": out},
              open(os.path.join(a.out, "s6_report.json"), "w"), indent=1)
    print("S6_DONE")


if __name__ == "__main__":
    main()
