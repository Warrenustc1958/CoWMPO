# Copyright 2026 The RLinf Authors.
"""Dense contact-timing score for rear-only rack candidates (v16). Builder side, no GT.

Evidence: the 3 rear probes (vert_x+080_*) re-exported with one public RGB frame per step over 160..340
(codex_rack_vertical_dense_v16) and tracked with the frozen v7b rule (rack_vertical_dense_v16_tracks).
These probes are FIT evidence (their Target first-contact was seen in the v11 check).
Per step, inside the rack ROI (frame-0 perception rack bbox + 20 px):
  roi_iou   observed v7b robot mask vs candidate's rendered robot mask
  dcy       settled robot-mask centroid row, sim - obs (px), at the last window frame
  stall     first step after ROI entry whose centroid stays within STALL_TOL px for STALL_WIN steps (same rule
            on obs and sim)
Candidates: v12, v15, v17, v18, v19 worlds; all keep the front slats and sides unchanged.
usage: python score_rack_dense_v16.py --sim <cand_result.json> <out.npz>   (render one candidate)
       python score_rack_dense_v16.py                                       (render all + score)
"""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
PLAN = R / "rack_vertical_dense_plan_v16"
TRACK = R / "rack_vertical_dense_v16_tracks/frames"
OUT = R / "rack_dense_score_v16"
EP = "libero_goal_t0_i0"
KEYS = ["vert_x+080_y-080", "vert_x+080_y+000", "vert_x+080_y+080"]
PAD, STALL_WIN, STALL_TOL = 20, 8, 0.5
CANDS = sorted((R / "rack_back_dev_v12").glob("cand??/result.json")) + \
    sorted((R / "rack_asym_dev_v15").glob("cand??/result.json")) + \
    sorted((R / "rack_post_dev_v17").glob("cand??/result.json")) + \
    sorted((R / "rack_post_dev_v18").glob("cand??/result.json")) + \
    sorted((R / "rack_post_dev_v19").glob("cand??/result.json")) + \
    sorted((R / "rack_post_dev_v20").glob("cand??/result.json")) + \
    sorted((R / "rack_post_dev_v21").glob("cand??/result.json"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def roi():
    with np.load(R / "visual_dev_bound_v1/dev3" / EP / "masks.npz") as a:
        m = a["rack"]
    ys, xs = np.nonzero(m)
    return max(0, ys.min() - PAD), min(256, ys.max() + 1 + PAD), max(0, xs.min() - PAD), min(256, xs.max() + 1 + PAD)


def window(key):
    p = json.loads((PLAN / key / "probe.json").read_text())
    s, e = p["dense_window"]
    return p, list(range(s, e + 1))


def sim(cand, out):
    from cowm.init.estimate_state import Scene
    from cowm.s6.plan_probe import apply_state
    y0, y1, x0, x1 = roi()
    res = json.loads(Path(cand).read_text())
    sc = Scene(res["world"])
    doc = json.loads((Path(res["states"]) / EP / "state.json").read_text())
    robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    arrs = {}
    for key in KEYS:
        probe, steps = window(key)
        sc.reset()
        apply_state(sc, doc["objects"], require_matched=True, robot=robot0)
        ms = []
        for step, act in enumerate(probe["actions"], 1):
            sc.env.step(np.asarray(act))
            if step in steps:
                ms.append(sc.robot_mask()[y0:y1, x0:x1])
            if step >= steps[-1]:
                break
        arrs[key] = np.stack(ms)
    sc.env.close()
    np.savez_compressed(out, **arrs)


def centroid(ms):
    """Per-frame robot-mask area and centroid row inside the ROI. (The lowest-row signal of the first draft was
    degenerate: a static robot part fixes it at one row for every frame, observed and simulated alike.)"""
    ys = np.mgrid[:ms.shape[1], :ms.shape[2]][0]
    area = ms.sum((1, 2))
    return (ms * ys).sum((1, 2)) / np.maximum(area, 1), area


def stall(cy, area, steps):
    """First step after ROI entry (area > 2x frame-0 area) whose centroid stays within STALL_TOL px for the next
    STALL_WIN steps. Same rule on observed and simulated masks."""
    ent = np.nonzero(area > 2 * area[0])[0]
    if not len(ent):
        return None
    for i in range(ent[0], len(cy) - STALL_WIN):
        if np.abs(cy[i:i + STALL_WIN + 1] - cy[i]).max() <= STALL_TOL:
            return steps[i]
    return steps[-1]


def observed():
    obs, files = {}, {}
    for key in KEYS:
        _, steps = window(key)
        ms = []
        for s in steps:
            fr = TRACK / key / f"{EP}__f{s:03d}"
            m = json.loads((fr / "association.json").read_text())
            if m["masks_sha256"] != sha(fr / "masks.npz") or m["objects"]["robot"]["status"] != "matched":
                raise ValueError(f"invalid observed evidence {fr}")
            files[str(fr / "masks.npz")] = m["masks_sha256"]
            with np.load(fr / "masks.npz") as a:
                ms.append(a["robot"])
        obs[key] = np.stack(ms)
    return obs, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", nargs=2)
    a = ap.parse_args()
    if a.sim:
        return sim(*a.sim)
    OUT.mkdir(exist_ok=True)
    (OUT / "sim").mkdir(exist_ok=True)

    def name(c):
        return f"{c.parent.parent.name}__{c.parent.name}"

    def run(c):
        dest = OUT / "sim" / f"{name(c)}.npz"
        if not dest.exists():
            with (OUT / "sim" / f"{name(c)}.log").open("w") as log:
                r = subprocess.run([sys.executable, __file__, "--sim", str(c), str(dest)],
                                   stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                raise RuntimeError((str(c), r.returncode))
        return dest

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(run, CANDS))
    print("SIM_DONE", len(CANDS), flush=True)
    if not TRACK.is_dir() or any(len(list((TRACK / k).glob("*/association.json"))) <
                                 len(json.loads((PLAN / k / "probe.json").read_text())["frames"]) for k in KEYS):
        print("TRACKS_NOT_READY", flush=True)
        return
    y0, y1, x0, x1 = roi()
    obs, files = observed()
    obs = {k: v[:, y0:y1, x0:x1] for k, v in obs.items()}
    ref = {}
    for k in KEYS:
        _, steps = window(k)
        cy, area = centroid(obs[k])
        ref[k] = {"cy": cy, "stall": stall(cy, area, steps), "steps": steps}
    rows = []
    for c in CANDS:
        res = json.loads(c.read_text())
        with np.load(OUT / "sim" / f"{name(c)}.npz") as z:
            pred = {k: z[k] for k in KEYS}
        per = {}
        for k in KEYS:
            o, p = obs[k], pred[k]
            u = (o | p).sum((1, 2))
            iou = np.where(u > 0, (o & p).sum((1, 2)) / np.maximum(u, 1), 1.0)
            cy, area = centroid(p)
            st = stall(cy, area, ref[k]["steps"])
            per[k] = {"roi_iou": round(float(iou.mean()), 4),
                      "settled_dcy_px": round(float(cy[-1] - ref[k]["cy"][-1]), 2),
                      "stall_sim": st, "stall_obs": ref[k]["stall"],
                      "stall_err": None if st is None or ref[k]["stall"] is None else abs(st - ref[k]["stall"])}
        row = {"cand": name(c), "params": {k: res[k] for k in ("e", "h", "h_pos", "b", "bar_y", "post_y", "bar_end_pos", "post_y_pos", "side", "frac") if k in res},
               "static_iou": round(res["static_iou"], 4), "per_probe": per,
               "roi_iou_mean": round(float(np.mean([v["roi_iou"] for v in per.values()])), 4),
               "dcy_abs_max": round(max(abs(v["settled_dcy_px"]) for v in per.values()), 2),
               "stall_err_max": max((v["stall_err"] if v["stall_err"] is not None else 999) for v in per.values())}
        # same weighting as v12 (static + 3 x response); response = dense ROI IoU
        row["loss"] = round(1 - row["static_iou"] + 3 * (1 - row["roi_iou_mean"]), 4)
        rows.append(row)
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError(f"input changed {p}")
    rows.sort(key=lambda r: r["loss"])
    (OUT / "score.json").write_text(json.dumps({
        "GT_inputs": False, "script_sha256": sha(__file__), "roi": [int(y0), int(y1), int(x0), int(x1)],
        "observed_stall": {k: ref[k]["stall"] for k in KEYS}, "n_observed_frames": len(files),
        "data_roles": "3 rear probes = fit evidence; acceptance needs a new unused probe set (x+040, x+120 rows)",
        "candidates": rows}, indent=1))
    for r in rows:
        print(json.dumps({k: r[k] for k in ("cand", "params", "static_iou", "roi_iou_mean", "dcy_abs_max",
                                             "stall_err_max", "loss")}), flush=True)
    print("DENSE_V16_DONE", flush=True)


if __name__ == "__main__":
    main()
