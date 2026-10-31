# Copyright 2026 The RLinf Authors.
"""Stove detent family v52 (v51 + knob Coulomb frictionloss: v51 policy_t7_i0 knob reaches the on-well then is dragged back by the gripper; the real knob stays on).
v51 header follows. Stove detent family v51 (Builder side; public RGB observations only, no GT). Codex's dial_detent family (10 worlds,
damping .02 fixed) best reaches 9/10 final states but onsets lag 9-12 steps (threshold rescoring cannot fix it), so
the knob dynamics are re-identified: well height x stiffness x damping = 36 worlds on the same rotating-dial world and
the same 10 public probes (query files copied from stove_dial_detent_v1), replayed with Codex's engine
(stove_axis_family_v2.worker + bistable_fixture). Selection offline over activation thresholds; rule (Codex's):
10/10 final states and every positive onset <= 8 steps; leave-one-probe-out reported.
usage: python stove_detent_v51.py --prepare | --index i | (run all)
"""
import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import bistable_fixture
import stove_axis_family_v2 as engine
from builder_batch import OUT as B, sha, write
from cowm.world.artifact import tree_sha

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
DEST = R / "stove_detent_v52"
SRC = B / "stove_dial_detent_v1"
WORLD0 = B / "stove_rotating_dial_v1/world0"
engine.DEST, engine.install = DEST, bistable_fixture.install
GRID = list(itertools.product([.6, .8], [.3, 1., 3.], [.005, .02], [0., .002, .005, .01, .02, .05]))
THRS = np.round(np.arange(.05, 1.4, .025), 3)


def prepare():
    DEST.mkdir(exist_ok=False)
    doc = json.loads((SRC / "freeze.json").read_text())
    for f in SRC.glob("*_query.json"):
        shutil.copy2(f, DEST / f.name)
    cands = []
    for i, (high, k, damp, fl) in enumerate(GRID):
        w = DEST / f"world{i:02d}"
        shutil.copytree(WORLD0, w)
        meta = json.loads((w / "world.json").read_text())
        obj = next(o for o in meta["objects"] if o["name"] == "stove")
        obj["joint_detent"] = {"stable_positions": [0., high], "stiffness": k, "potential": "nearest quadratic well"}
        p = w / obj["xml"]
        tree = ET.parse(p)
        tree.find(".//joint").set("damping", f"{damp}")
        tree.find(".//joint").set("frictionloss", f"{fl}")
        tree.write(p)
        meta.pop("world_sha256", None)
        meta["world_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        write(w / "world.json", meta)
        cands.append({"name": f"d52_{i:02d}", "frictionloss": fl, "world": str(w), "sha": tree_sha(w), "loss_Nm": 0., "activation": high / 2,
                      "high": high, "stiffness": k, "damping": damp})
    doc["candidates"] = cands
    doc["files"][__file__] = sha(__file__)
    doc["hypothesis"] = "re-identify detent well height, stiffness and damping on public red-onset transitions"
    write(DEST / "freeze.json", doc)


def score(trace, Q, thr):
    rows = []
    for t in trace:
        q = Q[t["key"]]
        pred = np.asarray(t["q"]) >= thr
        on = next((i for i in range(len(pred) - 1) if pred[i] and pred[i + 1]), None)
        ob = q["observed_onset"]
        rows.append({"key": t["key"], "final_correct": bool(pred[-1]) == bool(q["observed_on"][-1]),
                     "error": None if ob is None else (abs(on - ob) if on is not None else len(pred)),
                     "frame_errors": int(np.sum(pred[q["frames"]] != np.asarray(q["observed_on"])))})
    return rows


def rank(rows):
    errs = [r["error"] for r in rows if r["error"] is not None]
    return (sum(r["final_correct"] for r in rows), -(max(errs) if errs else 0), -sum(r["frame_errors"] for r in rows))


def run():
    fz = json.loads((DEST / "freeze.json").read_text())

    def job(i):
        with (DEST / f"{fz['candidates'][i]['name']}.log").open("w") as log:
            subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(len(fz["candidates"])) as pool:
        list(pool.map(job, range(len(fz["candidates"]))))
    Q = {f.name[:-11]: json.loads(f.read_text()) for f in DEST.glob("*_query.json")}
    traces = {c["name"]: json.loads((DEST / f"{c['name']}_replay.json").read_text()) for c in fz["candidates"]}
    table = []
    for c in fz["candidates"]:
        for thr in THRS:
            rows = score(traces[c["name"]], Q, thr)
            errs = [r["error"] for r in rows if r["error"] is not None]
            table.append({"cand": c["name"], "frictionloss": c["frictionloss"], "high": c["high"], "stiffness": c["stiffness"], "damping": c["damping"],
                          "thr": float(thr), "final": sum(r["final_correct"] for r in rows), "max_err": max(errs),
                          "frame_err": sum(r["frame_errors"] for r in rows),
                          "eligible": sum(r["final_correct"] for r in rows) == len(rows) and max(errs) <= 8, "rows": rows})
    table.sort(key=lambda t: (t["final"], -t["max_err"], -t["frame_err"]), reverse=True)
    keys = sorted(Q)
    loo = []
    for hold in keys:
        best = max(table, key=lambda t: rank([r for r in t["rows"] if r["key"] != hold]))
        r = next(x for x in best["rows"] if x["key"] == hold)
        loo.append({"held_out": hold, "chosen": [best["cand"], best["thr"]], **r})
    write(DEST / "selection.json", {"top": [{k: v for k, v in t.items() if k != "rows"} for t in table[:20]],
                                    "best_rows": table[0]["rows"], "loo": loo,
                                    "loo_final_correct": sum(x["final_correct"] for x in loo),
                                    "loo_onset_errors": [x["error"] for x in loo if x["error"] is not None],
                                    "n_eligible": sum(t["eligible"] for t in table), "GT_inputs": False})
    for t in table[:8]:
        print(json.dumps({k: v for k, v in t.items() if k != "rows"}), flush=True)
    print("BEST_ROWS", json.dumps(table[0]["rows"]), flush=True)
    print("LOO final", sum(x["final_correct"] for x in loo), "/", len(loo), "onset errs",
          [x["error"] for x in loo if x["error"] is not None], flush=True)
    print("N_ELIGIBLE", sum(t["eligible"] for t in table), flush=True)
    print("STOVE_V52_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.index is not None:
        engine.worker(a.index)
    else:
        run()
