"""E0-mini (plan §9.7-E): planted-defect detection / repair on top of the S6-selected world.

Defects (planted by the experimenter, values hidden from nothing else):
  param  : <material>.friction -> upper bound of its material prior
  struct : bowl_radius -> upper bound of its declared bounds (recompiled + S5 re-estimated)
Detection : held-out J(defect) > J(fitted) and the G2 gate verdict.
Repair    : rerun the S6 grid on FIT episodes starting from the defective world; repaired iff the selected
            value equals the fitted one (grid resolution) and held-out J returns to within 5% of J(fitted).
Every evaluation runs in a fresh subprocess (see s6_identify).
"""
import argparse
import json
import os
import subprocess
import sys

from cowm.s6.s6_identify import EVIDENCE, g2, objective
from cowm.world.compile_world import compile_world
from cowm.world.priors import MATERIALS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s6", required=True, help="s6_identify output dir (s6_report.json)")
    p.add_argument("--program", required=True)
    p.add_argument("--s5-masks", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--probe-masks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--material", default="ceramic")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = json.load(open(os.path.join(a.s6, "s6_report.json")))
    best = rep["variants"]["param+struct"]
    fit, held = rep["fit_episodes"], rep["heldout_episodes"]
    key = f"{a.material}.friction"
    mu_fit, r_fit = best["params"][key], best["params"]["bowl_radius"]
    w_fit, s_fit = best["world"], best["state"]
    J = lambda s: objective(s)
    res = {"fitted": {"params": best["params"], "heldout": best["heldout"], "J": best["heldout_J"]}}

    # --- parameter defect
    wc = json.load(open(os.path.join(w_fit, "world.json")))
    cls = wc["materials"][a.material]
    mu_bad = MATERIALS[cls]["friction"][2]
    hs = g2(a, w_fit, s_fit, held, mu_bad, os.path.join(a.out, "g2_param_defect.json"))
    grid = sorted({r["friction"] for r in rep["search"] if r["friction"] is not None} | {mu_bad})
    fits = {mu: J(g2(a, w_fit, s_fit, fit, mu, os.path.join(a.out, f"g2_param_repair_mu{mu}.json"))) for mu in grid}
    mu_rep = min(fits, key=fits.get)
    hr = g2(a, w_fit, s_fit, held, mu_rep, os.path.join(a.out, "g2_param_repaired.json"))
    res["param_defect"] = {"planted": {key: mu_bad}, "heldout": hs, "J": J(hs), "detected": J(hs) > res["fitted"]["J"],
                           "repair_grid_fitJ": fits, "repaired_value": mu_rep, "heldout_repaired": hr,
                           "J_repaired": J(hr),
                           "repaired": mu_rep == mu_fit and J(hr) <= 1.05 * res["fitted"]["J"]}

    # --- structural defect
    r_bad = float(wc["params"]["bowl_radius"]["bounds"][1])
    wd, sd = os.path.join(a.out, "world_struct_defect"), os.path.join(a.out, "state_struct_defect")
    compile_world(a.program, wd, {"bowl_radius": r_bad, key: mu_fit})
    subprocess.run([sys.executable, "-m", "cowm.init.estimate_state", "--world", wd, "--evidence", EVIDENCE,
                    "--masks", a.s5_masks, "--out", sd, "--episodes", ",".join(fit + held)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hs = g2(a, wd, sd, held, mu_fit, os.path.join(a.out, "g2_struct_defect.json"))
    cand = {r["radius"]: (r["world"], r["state"]) for r in rep["search"] if r["radius"] is not None}
    cand[r_bad] = (wd, sd)
    fits = {r: J(g2(a, w, s, fit, mu_fit, os.path.join(a.out, f"g2_struct_repair_r{r}.json")))
            for r, (w, s) in cand.items()}
    r_rep = min(fits, key=fits.get)
    hr = g2(a, cand[r_rep][0], cand[r_rep][1], held, mu_fit, os.path.join(a.out, "g2_struct_repaired.json"))
    res["struct_defect"] = {"planted": {"bowl_radius": r_bad}, "heldout": hs, "J": J(hs),
                            "detected": J(hs) > res["fitted"]["J"], "repair_grid_fitJ": fits,
                            "repaired_value": r_rep, "heldout_repaired": hr, "J_repaired": J(hr),
                            "repaired": r_rep == r_fit and J(hr) <= 1.05 * res["fitted"]["J"]}
    json.dump(res, open(os.path.join(a.out, "e0_mini.json"), "w"), indent=1)
    for k in ("param_defect", "struct_defect"):
        d = res[k]
        print("E0", k, "planted", d["planted"], "J_fitted", round(res["fitted"]["J"], 3), "J_defect", round(d["J"], 3),
              "detected", d["detected"], "repaired_value", d["repaired_value"], "J_repaired", round(d["J_repaired"], 3),
              "repaired", d["repaired"], flush=True)
    print("E0_DONE")


if __name__ == "__main__":
    main()
