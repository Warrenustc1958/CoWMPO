# Copyright 2026 The RLinf Authors.
"""Stove per-episode pose refinement v62 (Builder side; public RGB transitions only, no GT).
v61: every remaining stove failure is on episode t7_i0 (other episodes within tolerance), i.e. an episode-specific
initial-state error rather than knob dynamics. Here the dynamics are frozen at v61 r3_082 (thr .55) and only the
t7_i0 stove XY in the Builder state is refined on t7_i0's own 5 public probes (offset grid +-3 cm, 7x7).
usage: python stove_pose_v62.py            (all offsets in parallel + score)
       python stove_pose_v62.py --offset i (one offset)
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import latch_fixture
import stove_axis_family_v2 as engine
from builder_batch import write

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
FIT = R / "stove_detent_v60"                 # 22 public probes (queries + observations)
MODEL = R / "stove_detent_v61_r3"
NAME, THR = "r3_082", .55
OUT = R / "stove_pose_v62"
EP = "libero_goal_t7_i0"
OFFS = [(dx, dy) for dx in np.linspace(-.03, .03, 7) for dy in np.linspace(-.03, .03, 7)]


def keys():
    return sorted(f.name[:-11] for f in FIT.glob("*_query.json")
                  if json.loads(f.read_text())["episode"] == EP)


def one(i):
    dx, dy = OFFS[i]
    dest = OUT / f"off{i:02d}"
    dest.mkdir(parents=True)
    ks = keys()
    q0 = json.loads((FIT / f"{ks[0]}_query.json").read_text())
    st = json.loads(Path(q0["state"]).read_text())
    st["objects"]["stove"]["pos"][0] += float(dx)
    st["objects"]["stove"]["pos"][1] += float(dy)
    sp = dest / "state.json"
    write(sp, st)
    for k in ks:
        q = json.loads((FIT / f"{k}_query.json").read_text())
        q["state"] = str(sp)
        write(dest / f"{k}_query.json", q)
    c = next(x for x in json.loads((MODEL / "freeze.json").read_text())["candidates"] if x["name"] == NAME)
    write(dest / "freeze.json", {"candidates": [c], "queries": ks, "offset_m": [dx, dy], "GT_inputs": False})
    engine.DEST, engine.install = dest, latch_fixture.install
    engine.worker(0)


def score():
    rows = []
    for i, (dx, dy) in enumerate(OFFS):
        d = OUT / f"off{i:02d}"
        tr = {t["key"]: np.asarray(t["q"]) for t in json.loads((d / f"{NAME}_replay.json").read_text())}
        per, nf, errs = {}, 0, []
        for k, a in tr.items():
            q = json.loads((d / f"{k}_query.json").read_text())
            p = a >= THR
            on = next((j for j in range(len(p) - 1) if p[j] and p[j + 1]), None)
            fc = bool(p[-1]) == bool(q["observed_on"][-1])
            ob = q["observed_onset"]
            e = None if ob is None else (abs(on - ob) if on is not None else 999)
            per[k] = (fc, e)
            nf += fc
            if e is not None:
                errs.append(e)
        rows.append({"offset_cm": [round(dx * 100, 1), round(dy * 100, 1)], "final": nf, "n": len(tr),
                     "max_err": max(errs) if errs else 0, "per": per})
    rows.sort(key=lambda r: (-r["final"], r["max_err"], abs(r["offset_cm"][0]) + abs(r["offset_cm"][1])))
    write(OUT / "score.json", {"rows": rows, "threshold": THR, "model": NAME, "GT_inputs": False})
    for r in rows[:8]:
        print(json.dumps(r), flush=True)
    base = next(r for r in rows if r["offset_cm"] == [0.0, 0.0])
    print("BASELINE(0,0)", json.dumps(base), flush=True)
    print("POSE_V62_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int)
    a = ap.parse_args()
    if a.offset is not None:
        one(a.offset)
    else:
        OUT.mkdir(exist_ok=False)

        def job(i):
            with (OUT / f"off{i:02d}.log").open("w") as log:
                subprocess.run([sys.executable, __file__, "--offset", str(i)], stdout=log, stderr=subprocess.STDOUT,
                               check=True)
        with ThreadPoolExecutor(len(OFFS)) as pool:
            list(pool.map(job, range(len(OFFS))))
        score()
