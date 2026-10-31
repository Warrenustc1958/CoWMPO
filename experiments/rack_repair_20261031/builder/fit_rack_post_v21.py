# Copyright 2026 The RLinf Authors.
"""Rack rear-top repair v21 (extends v20): -y rear post (.45w, +h_n) and the bar's -y part fixed at the v19/v20
best; the bar's +y end, the +y rear post's y position and its height (may be LOWER than the slat model) vary.

Why (diag_rack_contact_v16 on vert_x+080_y+080, Builder side, no GT): with h=.02 the gripper hand lands on the
top bar (rack_g42) and the finger on the +y post top (rack_g40) and stops 2.6 px above the observed settle; with
h=.01 the finger slides past onto the upper slats (g30/g32) and settles closer (-1.6 px); even the unchanged
model (h=0, no bar) stops on the +y post top 1.7 px high. So at the +y rear corner the real top is lower or the
post is further inward than the model.
Front slats, sides and origin unchanged. Scores as v17 (static + v12 9-probe regression); dense rear score by
score_rack_dense_v16.py. No Target data.
usage: python fit_rack_post_v21.py [--index i]
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
OUT = R / "rack_post_dev_v21"
V12_FREEZE = R / "rack_back_dev_v12/freeze.json"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
H_N, BAR = 0.02, 0.008
PN, YN = .45, 0.          # -y post y (fraction of width); -y bar end (0 = post + .008)
# (+y post y as fraction of width, +y post raise (m, negative = lower), +y bar end as fraction of width)
GRID = [(py, hp, pe) for py in (.35, .25) for hp in (0., -.02, -.04) for pe in (.15, 0.)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_post(out_dir, width, depth, height, incline, py, hp, pe, pn=PN, yn=YN, density=600., friction=.7):
    """experimental_rack.build body; only the two rear posts and a top bar between them differ."""
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
    xr = .25 * depth                             # ---- rear posts + top bar (the only changed parts) ----
    hn_top, hp_top = height + H_N, height + hp
    body += part('box', [.008, .008, hn_top / 2], [xr, -pn * width, hn_top / 2])
    body += part('box', [.008, .008, hp_top / 2], [xr, py * width, hp_top / 2])
    y_neg = (yn * width / 2) if yn else (pn * width + .008)
    y_pos = pe * width
    body += part('box', [BAR, (y_pos + y_neg) / 2, .008], [xr, (y_pos - y_neg) / 2, hn_top - .008])
    return _save(out_dir, 'rack', _wrap('rack', '', body, hn_top, max(width, depth) / 2))


def worker(i):
    py, hp, pe = GRID[i]
    pn, yn = PN, YN
    base = json.loads((BASE / "result.json").read_text())
    W, D, H = base["dims_m"]
    inc = base["incline"]
    dest = OUT / f"cand{i:02d}"
    dest.mkdir()
    world = dest / "world"
    shutil.copytree(base["world"], world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(post_raise_neg=H_N, post_raise_pos=float(hp), post_bar=BAR, post_y_neg=float(pn),
                       bar_y_neg=float(yn), bar_end_pos=float(pe), post_y_pos=float(py))
    ob["top_z"] = float(H + H_N)
    build_post(world / "assets/rack", W, D, H, inc, py, hp, pe)
    meta.pop("world_sha256")
    meta["rack_post_v21"] = {"h_neg": H_N, "h_pos": hp, "b": BAR, "post_y_neg": pn, "bar_y_neg": yn, "bar_end_pos": pe,
                             "post_y_pos": py,
                             "base": str(BASE), "rule": "rear posts + top bar only; slats/front/sides/origin unchanged"}
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
    rep = {"index": i, "h": H_N, "h_pos": hp, "b": BAR, "post_y": pn, "bar_y": yn, "bar_end_pos": pe, "post_y_pos": py,
           "world": str(world), "world_tree_sha256": wh, "states": str(states),
           "static_iou": float(np.mean(static)), "robot_iou": float(np.mean([r["mean_iou"] for r in responses])),
           "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("POST_DONE", i, py, hp, pe, round(rep["static_iou"], 4), round(rep["robot_iou"], 4), round(rep["loss"], 4), flush=True)


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
        "queries": v12["queries"], "grid_postypos_hpos_barendpos": GRID, "h_neg": H_N, "bar": BAR,
        "post_y_neg": PN, "bar_y_neg": YN,
        "files": files, "script_sha256": sha(__file__), "GT_inputs": False, "derived_from": str(V12_FREEZE),
        "data_roles": "same fit evidence as v12 + dense rear probes (fit); acceptance needs NEW unused probes "
                      "(x+040 / x+120 rows)",
        "objective": "static IoU + v12 9-probe response here (front regression); dense rear score separately"}, indent=1))

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
    print("POST_V21_DONE", flush=True)


if __name__ == "__main__":
    main()
