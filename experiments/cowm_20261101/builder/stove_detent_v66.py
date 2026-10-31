# Copyright 2026 The RLinf Authors.
"""Stove v66 round (34 public probes incl. v65 twist-direction probes; one-way structure = latch_fixture fl_back_mult). v64: (v63 + knob hinge axis sign +z/-z as structural parameter). v63: (v61 + knob body offset kx,ky; fin angle +-0.8). v61: (env V61_DEST/SEED/CENTER/SCALE): shrinking local random search on 22 public probes. v60: 22 public probes (10 + the 12 v59 acceptance probes, observed re-derived from public RGB), wider local search around d58_067 (seed 60, 160). v58: local random search (seed 58) around v57 d57_033 (10/10 final, onset 17), adds k_on and damping (switching speed). v57: random-search identification (seed 57, 120 samples) of knob fin collision length/width/rest angle, joint frictionloss, k_off, s_up (k_on 3, s_down .1); same public probes/rule/LOO. v56: cylinder/short-fin knob collision x joint Coulomb frictionloss (blocks single-sided pushes, admits grasp twists); activation threshold limited to <= on-well. v55: the knob fin (9 cm lever, collision box) turns the knob under pure translational pushes (old/new probes, rot=0) that do not turn the real knob; collision fin shape x latch params. v54: (v53 latch never engaged: unlatched spring too stiff; now k_off weak / k_on strong). v53: (hysteresis detent via latch_fixture: v51/v52 midpoint two-well cannot keep weak pushes off and a dragged-back turn on).
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

import latch_fixture
import stove_axis_family_v2 as engine
from builder_batch import OUT as B, sha, write
from cowm.world.artifact import tree_sha

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
import os
DEST = R / os.environ["V61_DEST"]
ACC = R / "stove_accept_v59"
EV59 = Path("/data_cf/cowmpo/evidence/codex_stove_accept_v59")
SRC = B / "stove_dial_detent_v1"
WORLD0 = B / "stove_rotating_dial_v1/world0"
engine.DEST, engine.install = DEST, latch_fixture.install
SHAPES = {"fin_box": 'type="box" size="0.045 0.008 0.02"', "fin_l025": 'type="box" size="0.025 0.008 0.02"',
          "fin_l015": 'type="box" size="0.015 0.008 0.02"', "cyl_r020": 'type="cylinder" size="0.02 0.02"',
          "cyl_r030": 'type="cylinder" size="0.03 0.02"', "cyl_r045": 'type="cylinder" size="0.045 0.02"'}
_rng = np.random.default_rng(int(os.environ["V61_SEED"]))
_C = tuple(json.loads(os.environ["V61_CENTER"]))   # (L, W, th, fl, koff, up, kon, damp)
_S = float(os.environ["V61_SCALE"])
def _p(v, lo, hi):
    return float(np.clip(v * _rng.uniform(1 - _S, 1 + _S), lo, hi))
GRID = [tuple(_C[:10]) + (0., _C[10])] + [
    (_p(_C[0], .004, .045), _p(_C[1], .003, .012), float(np.clip(_C[2] + _rng.uniform(-.8, .8) * _S, -.8, .8)),
     _p(_C[3], .02, .8), _p(_C[4], .05, 2.), _p(_C[5], .2, .55),
     _p(_C[6], .5, 10.), _p(_C[7], .003, .1),
     float(np.clip(_C[8] + _rng.uniform(-.02, .02) * _S, -.02, .02)), float(np.clip(_C[9] + _rng.uniform(-.02, .02) * _S, -.02, .02)),
     0., float(np.exp(_rng.uniform(np.log(1.), np.log(20.)))))
    for _i in range(191)]
FIN = '<geom type="box" size="0.045 0.008 0.02" pos="0 0 0.02" group="0"'
THRS = np.round(np.arange(.05, 1.4, .025), 3)


def prepare():
    DEST.mkdir(exist_ok=False)
    doc = json.loads((SRC / "freeze.json").read_text())
    for f in SRC.glob("*_query.json"):
        shutil.copy2(f, DEST / f.name)
    from stove_axis_family_v2 import red_counts
    for k in json.loads((ACC / "freeze.json").read_text())["queries"]:
        q = json.loads((ACC / f"{k}_query.json").read_text())
        with np.load(EV59 / k / "public_stream.npz") as a:
            frames, steps = a["frames"], a["frame_idx"].tolist()
        assert steps == q["frames"]
        with np.load(B / "stove_policy_masks_v1" / q["episode"] / "masks.npz") as a:
            reds = red_counts(frames, a["stove"])
        on = [bool(r >= 20) for r in reds]
        q.update(red_pixels=reds, observed_on=on, observed_onset=next((steps[i] for i in range(len(on) - 1) if on[i] and on[i + 1]), None),
                 public_image_source=str(EV59 / k / "public_stream.npz"), purpose="fit evidence (v59 acceptance seen)")
        write(DEST / f"{k}_query.json", q)
        doc["queries"].append(k)
    P65, E65 = R / "stove_probe_v65", Path("/data_cf/cowmpo/evidence/codex_stove_probe_v65")
    for k in json.loads((P65 / "freeze.json").read_text())["queries"]:
        q = json.loads((P65 / f"{k}_query.json").read_text())
        with np.load(E65 / k / "public_stream.npz") as a:
            frames, steps = a["frames"], a["frame_idx"].tolist()
        assert steps == q["frames"]
        with np.load(B / "stove_policy_masks_v1" / q["episode"] / "masks.npz") as a:
            reds = red_counts(frames, a["stove"])
        on = [bool(r >= 20) for r in reds]
        q.update(red_pixels=reds, observed_on=on, observed_onset=next((steps[i] for i in range(len(on) - 1) if on[i] and on[i + 1]), None),
                 public_image_source=str(E65 / k / "public_stream.npz"), purpose="fit evidence (v65 twist probes)")
        write(DEST / f"{k}_query.json", q)
        doc["queries"].append(k)
    cands = []
    for i, (L, Wd, th, fl, koff, up, k, damp, kx, ky, flip, mult) in enumerate(GRID):
        high, down = .6, .1
        shape = f"box L{L:.3f} W{Wd:.3f} th{th:+.2f}"
        w = DEST / f"world{i:02d}"
        shutil.copytree(WORLD0, w)
        meta = json.loads((w / "world.json").read_text())
        obj = next(o for o in meta["objects"] if o["name"] == "stove")
        obj["joint_latch"] = {"stable_positions": [0., high], "s_up": up, "s_down": down, "stiffness": k, "k_off": koff, "k_on": k, "fl_back_mult": mult}
        p = w / obj["xml"]
        xml = p.read_text()
        assert xml.count(FIN) == 1
        xml = xml.replace(FIN, f'<geom type="box" size="{L:.4f} {Wd:.4f} 0.02" pos="0 0 0.02" euler="0 0 {th:.4f}" group="0"', 1)
        assert xml.count('<body name="knob" pos="-0.14446 0.0 0.024">') == 1
        p.write_text(xml.replace('<body name="knob" pos="-0.14446 0.0 0.024">', f'<body name="knob" pos="{-0.14446 + kx:.5f} {ky:.5f} 0.024">', 1))
        tree = ET.parse(p)
        tree.find(".//joint").set("damping", f"{damp}")
        tree.find(".//joint").set("frictionloss", f"{fl}")
        if flip:
            tree.find(".//joint").set("axis", "0 0 -1")
        tree.write(p)
        meta.pop("world_sha256", None)
        meta["world_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        write(w / "world.json", meta)
        cands.append({"name": f"{os.environ['V61_DEST'][-2:]}_{i:03d}", "params": [L, Wd, th, fl, koff, up, k, damp, kx, ky, mult], "knob_offset": [kx, ky], "axis_flip": bool(flip), "fl_back_mult": mult, "fin": [L, Wd, th], "frictionloss": fl, "shape": shape, "s_up": up, "s_down": down, "k_off": koff, "world": str(w), "sha": tree_sha(w), "loss_Nm": 0., "activation": high / 2,
                      "high": high, "stiffness": k, "damping": damp})
    doc["candidates"] = cands
    doc["files"][__file__] = sha(__file__)
    doc["files"][latch_fixture.__file__] = sha(latch_fixture.__file__)
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
    with ThreadPoolExecutor(64) as pool:
        list(pool.map(job, range(len(fz["candidates"]))))
    Q = {f.name[:-11]: json.loads(f.read_text()) for f in DEST.glob("*_query.json")}
    traces = {c["name"]: json.loads((DEST / f"{c['name']}_replay.json").read_text()) for c in fz["candidates"]}
    table = []
    for c in fz["candidates"]:
        for thr in THRS[THRS <= c["high"] + 1e-9]:
            rows = score(traces[c["name"]], Q, thr)
            errs = [r["error"] for r in rows if r["error"] is not None]
            table.append({"cand": c["name"], "frictionloss": c["frictionloss"], "shape": c["shape"], "k_off": c["k_off"], "s_up": c["s_up"], "s_down": c["s_down"], "high": c["high"], "stiffness": c["stiffness"], "damping": c["damping"],
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
    print("STOVE_V61_ROUND_DONE", flush=True)


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
