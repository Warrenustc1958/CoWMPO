# Copyright 2026 The RLinf Authors.
"""Rack -y rib repair v31: v29 cand10 (v23 cand03 + flat -y half-top plate dz .03, xf 0, ye -.10 W) plus a narrow
raised rib across the -y half (centre local x = cx, half-width hw, top at H + dzr, from the -y edge to local y = yer),
and the half-top plate's back edge at xb (D/2 = unchanged, or cut to .085 m).
Why: the transition-level public signal (diag_rack_contact_rgb_v31, deceleration onset dec0.5, same rule on public RGB
and on the model's robot mask) shows the model 4 steps late only at x+050_y-060 and x+060_y-040, while x+020/x+080/
x+100 y-040 are within 1 step: a narrow raised feature near local x .02-.05 on the -y half, ~3 cm above the plate.
x+120_y-080 is 6 steps early: the plate may be too long at the rear -y corner. Informed by v28 Target (x+050_y-060
fails) and by that public diagnostic (recorded in freeze.json). Selection: transition loss on public RGB +
guardrails, see score_rack_transition_v31.py. Index 0 (no rib, xb = D/2) must reproduce v29 cand10 exactly.
usage: python fit_rack_rib_v31.py [--index i]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import fit_rack_panel_v23 as v23
from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
BASE = R / "rack_scale_scan_dev_v8/scale00"
PREV = R / "rack_halftop_dev_v29/cand10"
OUT = R / "rack_rib_dev_v31"
V12_FREEZE = R / "rack_back_dev_v12/freeze.json"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
DZ0, X00, YE0 = .005, 0., .35                     # v23 cand03 rear top panel
PDZ, PXF, PYE = .03, 0., -.10                     # v29 cand10 half-top plate
RGBA = '0.45 0.3 0.2 1'
# (rib centre x, rib half-width x, rib top above H, rib +y end (local m), plate back edge (None = D/2))
GRID = [(None, None, None, None, None)] + [
    (cx, hw, dzr, yer, xb) for cx in (.025, .035, .045) for hw in (.01, .015) for dzr in (.05, .065)
    for yer in (-.035, -.02) for xb in (None, .085)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out_dir, width, depth, height, incline, cx, hw, dzr, yer, xb):
    orig = v23._wrap

    def wrap(name, assets, body, top, radius):
        xe, yl, yr, zt = depth / 2 if xb is None else xb, -width / 2, PYE * width, height + PDZ
        body += v23._part('box', [(xe - PXF) / 2, (yr - yl) / 2, .004], [(xe + PXF) / 2, (yr + yl) / 2, zt - .004],
                          RGBA, density=600., friction=.7)
        top2 = zt
        if cx is not None:      # rib sits on the plate top
            h = dzr - PDZ
            body += v23._part('box', [hw, (yer - yl) / 2, h / 2], [cx, (yer + yl) / 2, zt + h / 2],
                              RGBA, density=600., friction=.7)
            top2 = height + dzr
        return orig(name, assets, body, max(top, top2), radius)
    v23._wrap = wrap
    try:
        return v23.build_post(out_dir, width, depth, height, incline, DZ0, X00, YE0)
    finally:
        v23._wrap = orig


def worker(i):
    cx, hw, dzr, yer, xb = GRID[i]
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    inc = base["incline"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(base["world"], world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(post_raise_neg=v23.H_N, post_raise_pos=v23.HP, post_bar=v23.BAR, post_y_neg=v23.PN,
                       bar_y_neg=v23.YN, bar_end_pos=v23.PE, post_y_pos=v23.PY, panel_dz=DZ0, panel_x0=X00,
                       panel_y_end=YE0, halftop_dz=PDZ, halftop_xf=PXF, halftop_ye=PYE,
                       rib_cx=cx, rib_hw=hw, rib_dz=dzr, rib_ye=yer, halftop_xb=xb)
    ob["top_z"] = float(H + max(v23.H_N, DZ0, PDZ, dzr or 0.))
    build(world / "assets/rack", W, D, H, inc, cx, hw, dzr, yer, xb)
    meta.pop("world_sha256")
    meta["rack_rib_v31"] = {"base": str(BASE), "prev": str(PREV), "rib": [cx, hw, dzr, yer], "plate_xb": xb,
                            "rule": "v29 cand10 + narrow -y rib (+ optional plate back cut); rest unchanged"}
    meta["world_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    (world / "world.json").write_text(json.dumps(meta, indent=1))
    states = Path(base["states"])
    wh = tree_sha(world)
    scene = Scene(str(world))
    docs = {ep: json.loads((states / ep / "state.json").read_text()) for ep in EPS}
    static = []
    for ep in EPS:
        robot = json.loads((R / "robot_rgb_all_dev_v2" / ep / "robot.json").read_text())["robot"]
        scene.reset()
        apply_state(scene, docs[ep]["objects"], require_matched=True, robot=robot)
        k = scene.names.index("rack")
        with np.load(R / "visual_dev_bound_v1/dev3" / ep / "masks.npz") as a:
            target = a["rack"]
        static.append(iou(scene.seg() == k, target))
    robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    responses = []
    for q in json.loads((OUT / "freeze.json").read_text())["queries"]:
        scene.reset()
        apply_state(scene, docs[EP]["objects"], require_matched=True, robot=robot0)
        probe = json.loads(Path(q["source"]).read_text())
        vals = []
        for step, act in enumerate(probe["actions"], 1):
            scene.env.step(np.asarray(act))
            if step not in q["frames"]:
                continue
            fr = Path(q["video"]) / f"{EP}__f{step:03d}"
            m = json.loads((fr / "association.json").read_text())
            if m["masks_sha256"] != sha(fr / "masks.npz") or m["objects"]["robot"]["status"] != "matched":
                raise ValueError(f"invalid response evidence {fr}")
            with np.load(fr / "masks.npz") as a:
                vals.append(iou(scene.robot_mask(), a["robot"]))
            if step >= max(q["frames"]):
                break
        if len(vals) != len(q["frames"]):
            raise ValueError("incomplete contact response")
        responses.append({"key": q["key"], "group": q["group"], "mean_iou": float(np.mean(vals))})
    scene.env.close()
    verify_tree(world, wh)
    rep = {"index": i, "rib_cx": cx, "rib_hw": hw, "rib_dz": dzr, "rib_ye": yer, "halftop_xb": xb,
           "world": str(world), "world_tree_sha256": wh, "states": str(states),
           "static_iou": float(np.mean(static)), "robot_iou": float(np.mean([r["mean_iou"] for r in responses])),
           "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("RIB_DONE", i, cx, hw, dzr, yer, xb, round(rep["static_iou"], 4), round(rep["robot_iou"], 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.index is not None:
        return worker(a.index)
    OUT.mkdir(exist_ok=False)
    v12 = json.loads(V12_FREEZE.read_text())
    files = dict(v12["files"])
    for p, hsh in files.items():
        if sha(p) != hsh:
            raise ValueError(f"v12 evidence changed {p}")
    (OUT / "freeze.json").write_text(json.dumps({
        "queries": v12["queries"], "grid_cx_hw_dzr_yer_xb": GRID, "prev": str(PREV), "plate": [PDZ, PXF, PYE],
        "files": files, "script_sha256": sha(__file__), "v23_script_sha256": sha(v23.__file__), "GT_inputs": False,
        "derived_from": str(V12_FREEZE),
        "family_design_informed_by": "v28 Target outcome (x+050_y-060 fails) + public transition diagnostic "
                                     "diag_rack_contact_rgb_v31 (dec0.5); selection uses public RGB only",
        "data_roles": "fit evidence: all 27 public dense probes (v16 rear, v22/v24/v26/v28 positions); "
                      "acceptance on NEW unused positions",
        "objective": "static IoU + v12 9-probe response here; transition + dense scores separately"}, indent=1))

    def run(i):
        with (OUT / f"cand{i:02d}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((i, r.returncode))
        return json.loads((OUT / f"cand{i:02d}" / "result.json").read_text())

    with ThreadPoolExecutor(49) as pool:
        rows = list(pool.map(run, range(len(GRID))))
    for p, hsh in files.items():
        if sha(p) != hsh:
            raise ValueError(f"input changed {p}")
    ref = json.loads((PREV / "result.json").read_text())
    ident = {"static_iou": [rows[0]["static_iou"], ref["static_iou"]], "robot_iou": [rows[0]["robot_iou"], ref["robot_iou"]]}
    ok = all(abs(x - y) < 1e-6 for x, y in ident.values())
    brief = [{k: v for k, v in r.items() if k != "responses"} for r in rows]
    (OUT / "selection_v12_objective.json").write_text(json.dumps({"identity_check": ident, "identity_ok": ok,
                                                                  "candidates": brief}, indent=1))
    print("IDENTITY", "OK" if ok else "MISMATCH", ident, flush=True)
    print("RIB_V31_DONE", flush=True)


if __name__ == "__main__":
    main()
