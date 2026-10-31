# Copyright 2026 The RLinf Authors.
"""Rack asymmetric rear repair v15 (Builder side; no Target data).

v14 ROI evidence: raising the whole rear helps probe x+080/y+000 but hurts x+080/y+080. Hypothesis: the rear is
higher on one side only. Candidates keep the frozen v8 model (front, sides, origin, poses) and add rear posts +
a top rail only over a fraction `frac` of the width, starting from side `side` (-1: local -y edge, +1: mirror
control). Grid: side {-1} x frac {1/2, 2/3} x h {3, 6, 9 cm} x e {0, 3 cm}, plus mirror side {+1}, frac 1/2.
Score: 1 - static IoU + 3 x (1 - mean ROI robot-mask IoU over 9 probes) (ROI = rack bbox of frame-0 perception
mask + 20 px; v7b tracks). Also reports full-image response IoU. Writes rack_asym_dev_v15/candNN.
usage: python fit_rack_asym_v15.py [--index i]
"""
import argparse
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree
from cowm.world.templates import _part, _save, _wrap

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
BASE = R / "rack_scale_scan_dev_v8/scale00"
OUT = R / "rack_asym_dev_v15"
V12 = R / "rack_back_dev_v12"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
PAD = 20
GRID = ([(-1, f, h, e) for f in (0.5, 2 / 3) for h in (0.03, 0.06, 0.09) for e in (0.0, 0.03)]
        + [(1, 0.5, h, e) for h in (0.03, 0.06, 0.09) for e in (0.0, 0.03)])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out_dir, width, depth, height, incline, side, frac, h, e, density=600., friction=.7):
    rgba = '0.45 0.3 0.2 1'

    def part(kind, size, pos, euler=None):
        return _part(kind, size, pos, rgba, euler=euler, density=density, friction=friction)
    th = .008
    slope, co = math.sin(incline), math.cos(incline)
    upper = height - .45 * depth * abs(slope) - th
    lower = max(.05, .45 * upper)
    body = part('box', [depth / 2, width / 2, .004], [0, 0, .004])           # original model, unchanged
    for level in [lower, upper]:
        for j in range(8):
            x = (-.45 + .9 * j / 7) * depth
            body += part('box', [.035 * depth, width / 2, th / 2], [co * x, 0, level - slope * x], euler=f'0 {incline} 0')
        body += part('box', [.012, width / 2, .006], [0, 0, level - .012])
    for s in [-1, 1]:
        body += part('box', [.008, .008, height / 2], [.25 * depth, s * .35 * width, height / 2])
    # ---- one-sided rear additions ----
    xb = depth / 2 + e - .008
    hb = height + h
    y_edge = side * width / 2
    y_inner = y_edge - side * frac * width
    yc, half = (y_edge + y_inner) / 2, frac * width / 2
    for y in (y_edge - side * .008, y_inner):
        body += part('box', [.008, .008, hb / 2], [xb, y, hb / 2])
    body += part('box', [.008, half, .008], [xb, yc, hb - .008])
    if e > 0:
        body += part('box', [e / 2, half, .004], [depth / 2 + e / 2, yc, .004])
    return _save(out_dir, 'rack', _wrap('rack', '', body, hb, max(width, depth + e) / 2))


def roi_box():
    with np.load(R / "visual_dev_bound_v1/dev3" / EP / "masks.npz") as a:
        m = a["rack"]
    ys, xs = np.nonzero(m)
    return max(0, ys.min() - PAD), min(256, ys.max() + 1 + PAD), max(0, xs.min() - PAD), min(256, xs.max() + 1 + PAD)


def worker(i):
    side, frac, h, e = GRID[i]
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(base["world"], world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(rear_side=side, rear_frac=float(frac), back_raise=float(h), back_ext=float(e))
    ob["top_z"] = float(H + h)
    build(world / "assets/rack", W, D, H, base["incline"], side, frac, h, e)
    meta.pop("world_sha256")
    meta["rack_asym_v15"] = {"side": side, "frac": frac, "h": h, "e": e, "base": str(BASE)}
    meta["world_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    (world / "world.json").write_text(json.dumps(meta, indent=1))
    states = Path(base["states"])
    wh = tree_sha(world)
    sc = Scene(str(world))
    docs = {ep: json.loads((states / ep / "state.json").read_text()) for ep in EPS}
    static = []
    for ep in EPS:
        rob = json.loads((R / "robot_rgb_all_dev_v2" / ep / "robot.json").read_text())["robot"]
        sc.reset()
        apply_state(sc, docs[ep]["objects"], require_matched=True, robot=rob)
        with np.load(R / "visual_dev_bound_v1/dev3" / ep / "masks.npz") as a:
            static.append(iou(sc.seg() == sc.names.index("rack"), a["rack"]))
    y0, y1, x0, x1 = roi_box()
    robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    per, roi_all, full_all = {}, [], []
    for q in json.loads((V12 / "freeze.json").read_text())["queries"]:
        probe = json.loads(Path(q["source"]).read_text())
        ph = list(probe["phases"])
        k = ph.index("contact")
        start, end = probe["phases"][ph[k - 1]] - 20, probe["phases"]["contact"]
        frames = [s for s in probe["frames"] if start < s <= end]
        sc.reset()
        apply_state(sc, docs[EP]["objects"], require_matched=True, robot=robot0)
        v = []
        for step, act in enumerate(probe["actions"], 1):
            sc.env.step(np.asarray(act))
            if step in frames:
                fr = Path(q["video"]) / f"{EP}__f{step:03d}"
                if json.loads((fr / "association.json").read_text())["masks_sha256"] != sha(fr / "masks.npz"):
                    raise ValueError("track hash")
                with np.load(fr / "masks.npz") as a:
                    obs = a["robot"]
                pred = sc.robot_mask()
                o, p = obs[y0:y1, x0:x1], pred[y0:y1, x0:x1]
                u = (o | p).sum()
                v.append(float((o & p).sum() / u) if u else 1.0)
                full_all.append(iou(pred, obs))
            if step >= end:
                break
        per[q["key"]] = round(float(np.mean(v)), 4)
        roi_all += v
    sc.env.close()
    verify_tree(world, wh)
    rep = {"index": i, "side": side, "frac": frac, "h": h, "e": e, "world": str(world), "world_tree_sha256": wh,
           "states": str(states), "static_iou": float(np.mean(static)), "roi_iou": float(np.mean(roi_all)),
           "full_iou": float(np.mean(full_all)), "per_probe_roi": per}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["roi_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("ASYM_DONE", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()
                                   if k in ("index", "side", "frac", "h", "e", "static_iou", "roi_iou", "loss")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.index is not None:
        return worker(a.index)
    OUT.mkdir(exist_ok=False)
    (OUT / "freeze.json").write_text(json.dumps({
        "grid_side_frac_h_e": GRID, "base": str(BASE), "queries_from": str(V12 / "freeze.json"), "roi_pad": PAD,
        "script_sha256": sha(__file__), "GT_inputs": False,
        "objective": "1 - static IoU + 3 x (1 - ROI robot IoU), 9 probes, v7b tracks; side +1 = mirror control"},
        indent=1))
    baseline = json.loads((V12 / "roi_v14/cand00.json").read_text())

    def run(i):
        with (OUT / f"cand{i:02d}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((i, r.returncode))
        return json.loads((OUT / f"cand{i:02d}" / "result.json").read_text())

    with ThreadPoolExecutor(9) as pool:
        rows = list(pool.map(run, range(len(GRID))))
    brief = [{k: v for k, v in r.items() if k != "per_probe_roi"} for r in rows]
    base_loss = 1 - 0.7219534140944291 + 3 * (1 - baseline["roi_iou_mean"])
    best = min(brief, key=lambda r: r["loss"])
    (OUT / "selection.json").write_text(json.dumps({"baseline_loss_same_objective": base_loss, "best": best,
                                                    "candidates": brief}, indent=1))
    print("ASYM_SELECTED", json.dumps(best), "baseline_loss", round(base_loss, 4), flush=True)


if __name__ == "__main__":
    main()
