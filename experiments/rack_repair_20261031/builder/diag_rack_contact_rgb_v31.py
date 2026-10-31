# Copyright 2026 The RLinf Authors.
"""Builder diagnostic v31 (public RGB + Builder model only, no GT, no Target numbers): a transition-level contact
signal. Acceptance is judged on first-contact step, but the fit so far used ROI mask IoU, which saturates (v30 edge
variants leave the -y edge probes' IoU unchanged). Here, per public probe:
  obs_stop   first step after the descent motion peak where full-frame RGB change (vs previous frame, >30 summed)
             stays < FRAC x peak for HOLD steps (the arm is blocked)
  sim_stop   same rule on the candidate's rendered robot-mask change
  sim_first  the candidate's own physical gripper-rack first contact step
sim_stop - sim_first calibrates the rule on the model; obs_stop - (sim_stop - sim_first) estimates the observed first
contact. Only Builder-side quantities. Writes R/diag_rack_contact_rgb_v31/<cand>.json
usage: python diag_rack_contact_rgb_v31.py <cand_result.json> [<cand_result.json> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
EV = Path("/data_cf/cowmpo/evidence")
SETS = {"rear16": ("rack_vertical_dense_plan_v16", "codex_rack_vertical_dense_v16"),
        "acc23": ("rack_accept_dense_plan_v23", "codex_rack_accept_dense_v23"),
        "acc25": ("rack_accept_dense_plan_v25", "codex_rack_accept_dense_v25"),
        "acc27": ("rack_accept_dense_plan_v27", "codex_rack_accept_dense_v27"),
        "acc30": ("rack_accept_dense_plan_v30", "codex_rack_accept_dense_v30")}
EP = "libero_goal_t0_i0"
LO, HI, FRAC, HOLD = 196, 260, .15, 3
OUT = R / "diag_rack_contact_rgb_v31"


def stop(ch, steps):
    ch = np.asarray(ch, float)
    pk = int(np.argmax(ch))
    for i in range(pk, len(ch) - HOLD):
        if (ch[i:i + HOLD] < FRAC * ch[pk]).all():
            return steps[i]
    return None


def decel(ch, steps, frac):
    """first step after the descent peak where the change drops below frac x peak (blocked moment; works while sliding)"""
    ch = np.asarray(ch, float)
    pk = int(np.argmax(ch))
    for i in range(pk + 1, len(ch)):
        if ch[i] < frac * ch[pk]:
            return steps[i]
    return None


RULES = (.3, .5, .7)


def probes():
    for s, (plan, ev) in SETS.items():
        for d in sorted((R / plan).iterdir()):
            if (d / "probe.json").exists():
                yield s, d.name, d / "probe.json", EV / ev / d.name


def obs_change(evdir, steps):
    prev, out = None, []
    for t in [steps[0] - 1] + steps:
        im = np.asarray(Image.open(evdir / f"{EP}__f{t:03d}" / "frame0.png")).astype(int)
        if prev is not None:
            out.append(int((np.abs(im - prev).sum(2) > 30).sum()))
        prev = im
    return out


def run(cand):
    res = json.loads(Path(cand).read_text())
    sc = Scene(res["world"])
    m = sc.env.sim.model
    k = sc.names.index("rack")
    grip = np.array([(m.body_id2name(int(m.geom_bodyid[g])) or "").startswith("gripper0") for g in range(m.ngeom)])
    doc = json.loads((Path(res["states"]) / EP / "state.json").read_text())
    robot = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    steps = list(range(LO, HI + 1))
    rows = {}
    for s, key, pj, evdir in probes():
        probe = json.loads(pj.read_text())
        sc.reset()
        apply_state(sc, doc["objects"], require_matched=True, robot=robot)
        d = sc.env.sim.data
        first, prev, sch = None, None, []
        for step, act in enumerate(probe["actions"], 1):
            sc.env.step(np.asarray(act))
            if first is None and any((grip[int(c.geom1)] and sc.geom2obj[int(c.geom2)] == k) or
                                     (grip[int(c.geom2)] and sc.geom2obj[int(c.geom1)] == k)
                                     for c in d.contact[:d.ncon]):
                first = step
            if step >= LO - 1:
                mk = sc.robot_mask()
                if prev is not None:
                    sch.append(int((mk ^ prev).sum()))
                prev = mk
            if step >= HI:
                break
        och = obs_change(evdir, steps)
        o, si = stop(och, steps), stop(sch, steps)
        rows[key] = {"set": s, "sim_first": first, "sim_stop": si, "obs_stop": o,
                     "obs_first_est": None if None in (o, si, first) else o - (si - first),
                     "sim_change": sch, "obs_change": och}
        for f in RULES:
            rows[key][f"sim_dec{f}"] = decel(sch, steps, f)
            rows[key][f"obs_dec{f}"] = decel(och, steps, f)
    sc.env.close()
    OUT.mkdir(exist_ok=True)
    name = f"{Path(cand).parent.parent.name}__{Path(cand).parent.name}"
    (OUT / f"{name}.json").write_text(json.dumps({"cand": cand, "rule": [LO, HI, FRAC, HOLD], "GT_inputs": False,
                                                  "Target_numbers_read": False, "probes": rows}, indent=1))
    print("==", name)
    for key, r in rows.items():
        print("  %-22s sim_first %s | stop sim %s obs %s | " % (key, r["sim_first"], r["sim_stop"], r["obs_stop"]) +
              " ".join("dec%.1f sim %s obs %s" % (f, r[f"sim_dec{f}"], r[f"obs_dec{f}"]) for f in RULES), flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(c)
