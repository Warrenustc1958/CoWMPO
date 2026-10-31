# Copyright 2026 The RLinf Authors.
"""Independent stove acceptance v59 (Builder side; public RGB only, no GT).
Frozen model: stove_detent_v58 d58_067 (knob fin collision 0.8 x 0.5 cm at -0.35 rad, frictionloss .125, latch
k_off .69 / s_up .325 / s_down .1 / k_on 3.33, damping .060), activation threshold .575; selected on 10 public probes
(10/10 final, onset <= 5; LOO 10/10).
Probes: the 12 Codex active-contrast queries (stove_active_contrast_v1) that were never executed on Target and never
used in v51-v58. Rule (fixed here, before any Target execution): every probe's final on/off correct, and for every
probe whose public RGB turns red, |model onset - observed onset| <= 8 steps. Observed on = red pixels >= 20 inside the
public stove mask (same as Codex score_stove_active.py).
usage: python stove_accept_v59.py --prepare | --replay | --score
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import latch_fixture
import stove_axis_family_v2 as engine
from builder_batch import OUT as B, sha, write
from stove_axis_family_v2 import red_counts

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
DEST = R / "stove_accept_v59"
SRC = B / "stove_active_contrast_v1"
E = Path("/data_cf/cowmpo/evidence/codex_stove_accept_v59")
MODEL = R / "stove_detent_v58"
NAME, THR = "d58_067", .575
engine.DEST, engine.install = DEST, latch_fixture.install
RULE = {"final_correct": "all probes", "onset_tol_steps": 8, "observed_on": "red pixels >= 20 in public stove mask"}


def onset(on, steps):
    on = list(on)
    return next((steps[i] for i in range(len(on) - 1) if on[i] and on[i + 1]), None)


def prepare():
    used = {r["key"] for r in json.loads((SRC / "selected.json").read_text())["queries"]}
    allq = sorted(json.loads((SRC / "freeze.json").read_text())["queries"])
    keys = [k for k in allq if k not in used]
    assert len(keys) == 12, keys
    DEST.mkdir(exist_ok=False)
    for k in keys:
        shutil.copy2(SRC / f"{k}_query.json", DEST / f"{k}_query.json")
    c = next(x for x in json.loads((MODEL / "freeze.json").read_text())["candidates"] if x["name"] == NAME)
    write(DEST / "freeze.json", {"candidates": [c], "queries": keys, "threshold": THR, "rule": RULE,
                                 "selection_source": str(MODEL / "selection.json"),
                                 "query_sha256": {k: sha(DEST / f"{k}_query.json") for k in keys},
                                 "script_sha256": sha(__file__), "fixture_sha256": sha(latch_fixture.__file__),
                                 "GT_inputs": False, "Target_queries_executed": 0,
                                 "data_roles": "acceptance-only until scored; never used for v51-v58 fitting"})
    print("PREPARED", keys, flush=True)


def score():
    fz = json.loads((DEST / "freeze.json").read_text())
    tr = {r["key"]: np.asarray(r["q"]) for r in json.loads((DEST / f"{NAME}_replay.json").read_text())}
    rows = []
    for k in fz["queries"]:
        q = json.loads((DEST / f"{k}_query.json").read_text())
        p = E / k / "public_stream.npz"
        meta = json.loads((p.parent / "query.json").read_text())
        if meta["query_sha256"] != fz["query_sha256"][k]:
            raise ValueError("query mismatch")
        with np.load(p) as a:
            frames, steps = a["frames"], a["frame_idx"].tolist()
        with np.load(B / "stove_policy_masks_v1" / q["episode"] / "masks.npz") as a:
            stove = a["stove"]
        obs = np.asarray(red_counts(frames, stove)) >= 20
        qq = tr[k]
        model_on = qq >= THR
        o_on, m_on = onset(obs, steps), onset(model_on, list(range(len(qq))))
        fc = bool(model_on[-1]) == bool(obs[-1])
        err = None if o_on is None else (abs(m_on - o_on) if m_on is not None else 999)
        rows.append({"key": k, "observed_final": bool(obs[-1]), "model_final": bool(model_on[-1]), "final_correct": fc,
                     "observed_onset": o_on, "model_onset": m_on, "onset_err": err,
                     "pass": fc and (err is None or err <= RULE["onset_tol_steps"]),
                     "model_max_q": round(float(qq.max()), 3), "prefix_exact_frames": meta["prefix_exact_frames"]})
        print(json.dumps(rows[-1]), flush=True)
    summ = {"rule": RULE, "n": len(rows), "passed": sum(r["pass"] for r in rows),
            "n_observed_on": sum(r["observed_final"] for r in rows), "rows": rows,
            "decision": "ACCEPT" if all(r["pass"] for r in rows) else "REJECT", "GT_inputs": False}
    write(DEST / "acceptance.json", summ)
    print("STOVE_ACCEPT_V59", summ["passed"], "/", summ["n"], "observed_on", summ["n_observed_on"], summ["decision"],
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.replay:
        engine.worker(0)
    elif a.score:
        score()
