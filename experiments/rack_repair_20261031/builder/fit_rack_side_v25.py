# Copyright 2026 The RLinf Authors.
"""Rack -y rear repair v25: v23 cand03 (rear posts + bar + rear top panel dz .005) plus a -y rear board sloping
down towards the back: top at H + hf at local x = XF and H + hb at the back edge (x = depth/2), from the -y edge to
y = ye*width. Everything else unchanged.

Why: the v24 independent acceptance REJECTED v23 cand03 (4/6); both failures are on the -y side (x+060, x+100).
Builder diagnostic diag_rack_first_contact_v25: the model's first contact there is the flat rear top panel
(z ~ .289 m above the table) at every position. The family (raised, front-high -y board) was designed from that
rejection, i.e. it is informed by the v24 Target first-contact outcome; the v24 positions are fit evidence now.
Selection uses public RGB only: static first-frame IoU + v12 9-probe regression here, dense score over 15 public
probes in score_rack_dense_v25.py. A new unused probe set is required for acceptance.
usage: python fit_rack_side_v25.py [--index i]
"""
import argparse
import hashlib
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
PREV = R / "rack_panel_dev_v23/cand03"
OUT = R / "rack_side_dev_v25"
V12_FREEZE = R / "rack_back_dev_v12/freeze.json"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
H_N, BAR = 0.02, 0.008
PN, YN, PY, HP, PE = .45, 0., .25, 0., 0.      # v21 cand07 posts/bar
DZ, X0, YE = .005, 0., .35                     # v23 cand03 rear top panel
XF = .02                                        # -y board front edge (m, rack-local x)
# -y board: (front top above H (m), back top above H (m), +y end as fraction of width)
GRID = [(hf, hb, ye) for hf in (.02, .035, .05) for hb in (.005, .015) for ye in (-.15, .05)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out_dir, width, depth, height, incline, hf, hb, ye, density=600., friction=.7):
    """fit_rack_panel_v23.build_post at cand03 values plus the sloped -y board."""
    rgba = '0.45 0.3 0.2 1'

    def part(kind, size, pos, euler=None):
        return _part(kind, size, pos, rgba, euler=euler, density=density, friction=friction)
    thickness = .008
    slope, co = math.sin(incline), math.cos(incline)
    upper = height - .45 * depth * abs(slope) - thickness
    lower = max(.05, .45 * upper)
    body = part('box', [depth / 2, width / 2, .004], [0, 0, .004])
    for level in [lower, upper]:
        for j in range(8):
            x = (-.45 + .9 * j / 7) * depth
            body += part('box', [.035 * depth, width / 2, thickness / 2], [co * x, 0, level - slope * x], euler=f'0 {incline} 0')
        body += part('box', [.012, width / 2, .006], [0, 0, level - .012])
    xr = .25 * depth                             # ---- v21 rear posts + top bar ----
    hn_top, hp_top = height + H_N, height + HP
    body += part('box', [.008, .008, hn_top / 2], [xr, -PN * width, hn_top / 2])
    body += part('box', [.008, .008, hp_top / 2], [xr, PY * width, hp_top / 2])
    y_neg, y_pos = PN * width + .008, PE * width
    body += part('box', [BAR, (y_pos + y_neg) / 2, .008], [xr, (y_pos - y_neg) / 2, hn_top - .008])
    xf, xb, yl, yr, zt = X0 * depth, depth / 2, -width / 2, YE * width, height + DZ    # ---- v23 rear top panel ----
    body += part('box', [(xb - xf) / 2, (yr - yl) / 2, .004], [(xb + xf) / 2, (yr + yl) / 2, zt - .004])
    za, zb = height + hf, height + hb            # ---- v25 sloped -y board (the only new part) ----
    th = math.atan2(za - zb, xb - XF)            # +pitch about y lowers z with x (same convention as the slats)
    half = math.hypot(xb - XF, za - zb) / 2
    cx, cz = (XF + xb) / 2 - .004 * math.sin(th), (za + zb) / 2 - .004 * math.cos(th)
    ys = ye * width
    body += part('box', [half, (ys - yl) / 2, .004], [cx, (ys + yl) / 2, cz], euler=f'0 {th} 0')
    top = max(hn_top, zt, za)
    return _save(out_dir, 'rack', _wrap('rack', '', body, top, max(width, depth) / 2))


def worker(i):
    hf, hb, ye = GRID[i]
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    inc = base["incline"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(base["world"], world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(post_raise_neg=H_N, post_raise_pos=HP, post_bar=BAR, post_y_neg=PN, bar_y_neg=YN,
                       bar_end_pos=PE, post_y_pos=PY, panel_dz=DZ, panel_x0=X0, panel_y_end=YE,
                       side_hf=float(hf), side_hb=float(hb), side_ye=float(ye), side_xf=XF)
    ob["top_z"] = float(H + max(H_N, DZ, hf))
    build(world / "assets/rack", W, D, H, inc, hf, hb, ye)
    meta.pop("world_sha256")
    meta["rack_side_v25"] = {"base": str(BASE), "prev": str(PREV), "side_hf": hf, "side_hb": hb, "side_ye": ye,
                             "side_xf": XF, "rule": "v23 cand03 + sloped -y rear board; everything else unchanged"}
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
    rep = {"index": i, "side_hf": hf, "side_hb": hb, "side_ye": ye, "side_xf": XF, "panel_dz": DZ, "h": H_N,
           "world": str(world), "world_tree_sha256": wh, "states": str(states),
           "static_iou": float(np.mean(static)), "robot_iou": float(np.mean([r["mean_iou"] for r in responses])),
           "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("SIDE_DONE", i, hf, hb, ye, round(rep["static_iou"], 4), round(rep["robot_iou"], 4), round(rep["loss"], 4),
          flush=True)


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
        "queries": v12["queries"], "grid_hf_hb_ye": GRID, "side_xf": XF, "prev": str(PREV),
        "files": files, "script_sha256": sha(__file__), "GT_inputs": False, "derived_from": str(V12_FREEZE),
        "family_design_informed_by": "v24 Target first-contact outcome (-y side earlier) + Builder diagnostic "
                                     "diag_rack_first_contact_v25.json; selection itself uses public RGB only",
        "data_roles": "fit evidence: v12 9 probes, dense v16 rear, dense v23 (v22 positions), dense v25 (v24 "
                      "positions); acceptance needs a NEW unused probe set",
        "objective": "static IoU + v12 9-probe response here (front regression); dense score separately"}, indent=1))

    def run(i):
        with (OUT / f"cand{i:02d}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((i, r.returncode))
        return json.loads((OUT / f"cand{i:02d}" / "result.json").read_text())

    with ThreadPoolExecutor(12) as pool:
        rows = list(pool.map(run, range(len(GRID))))
    for p, hsh in files.items():
        if sha(p) != hsh:
            raise ValueError(f"input changed {p}")
    brief = [{k: v for k, v in r.items() if k != "responses"} for r in rows]
    (OUT / "selection_v12_objective.json").write_text(json.dumps({"candidates": brief}, indent=1))
    print("SIDE_V25_DONE", flush=True)


if __name__ == "__main__":
    main()
