# Copyright 2026 The RLinf Authors.
"""Dense contact score for rack candidates over four public probe sets (v30: 27 probes incl. v28 positions, -y edge candidates; + loss_ex_ym060 sensitivity).
Builder side, no GT.

Probe sets (all FIT evidence):
  acc27   6 positions of the rejected v26 acceptance (codex_rack_accept_dense_v27, frozen v7b tracks
          rack_accept_dense_v27_tracks); their Target first-contact numbers are NOT read here
  acc25   6 positions of the rejected v24 acceptance (codex_rack_accept_dense_v25, frozen v7b tracks
          rack_accept_dense_v25_tracks); their Target first-contact numbers are NOT read here
  rear16  3 rear probes vert_x+080_*  (codex_rack_vertical_dense_v16, frozen v7b tracks rack_vertical_dense_v16_tracks)
  acc23   6 positions of the rejected v22 acceptance (codex_rack_accept_dense_v23, frozen v7b tracks
          rack_accept_dense_v23_tracks); their Target first-contact numbers are NOT read here
Per probe, inside the rack ROI (frame-0 perception rack bbox + 20 px), over its dense window:
  roi_iou   observed v7b robot mask vs candidate's rendered robot mask (mean over window steps)
  dcy       settled robot-mask centroid row, sim - obs (px), at the last window frame
  stall     first step after ROI entry whose centroid stays within STALL_TOL px for STALL_WIN steps (same rule on
            obs and sim)
loss = 1 - static IoU + 3 x (1 - mean roi_iou over all 27 probes)   (same weighting as v12/v16/v23)
usage: python score_rack_dense_v30.py --sim <cand_result.json> <out.npz>   (render one candidate)
       python score_rack_dense_v30.py                                       (render all + score)
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
SETS = {"rear16": (R / "rack_vertical_dense_plan_v16", R / "rack_vertical_dense_v16_tracks/frames",
                   ["vert_x+080_y-080", "vert_x+080_y+000", "vert_x+080_y+080"]),
        "acc23": (R / "rack_accept_dense_plan_v23", R / "rack_accept_dense_v23_tracks/frames",
                  ["acc_x+040_y+000", "acc_x+040_y+080", "acc_x+040_y-080", "acc_x+120_y+000", "acc_x+120_y+080",
                   "acc_x+120_y-080"]),
        "acc25": (R / "rack_accept_dense_plan_v25", R / "rack_accept_dense_v25_tracks/frames",
                  ["acc24_x+060_y+040", "acc24_x+060_y-040", "acc24_x+080_y+040", "acc24_x+080_y-040",
                   "acc24_x+100_y+040", "acc24_x+100_y-040"]),
        "acc27": (R / "rack_accept_dense_plan_v27", R / "rack_accept_dense_v27_tracks/frames",
                  ["acc26_x+020_y+040", "acc26_x+020_y-040", "acc26_x+070_y-060", "acc26_x+090_y-060", "acc26_x+110_y+040", "acc26_x+110_y-040"]),
        "acc30": (R / "rack_accept_dense_plan_v30", R / "rack_accept_dense_v30_tracks/frames",
                  ["acc28_x+030_y+020", "acc28_x+030_y-020", "acc28_x+050_y-060", "acc28_x+100_y-020", "acc28_x+130_y+020", "acc28_x+130_y-020"])}
PROBES = [(s, k) for s, (_, _, ks) in SETS.items() for k in ks]
OUT = R / "rack_dense_score_v30"
EP = "libero_goal_t0_i0"
PAD, STALL_WIN, STALL_TOL = 20, 8, 0.5
YM060 = ("acc26_x+070_y-060", "acc26_x+090_y-060", "acc28_x+050_y-060")   # slide off the -y edge (diag_rack_ym060_v28); sensitivity only
CANDS = [R / "rack_back_dev_v12/cand00/result.json", R / "rack_post_dev_v21/cand07/result.json",
         R / "rack_panel_dev_v23/cand03/result.json", R / "rack_side_dev_v25/cand01/result.json",
         R / "rack_halftop_dev_v29/cand10/result.json"] + sorted((R / "rack_edge_dev_v30").glob("cand??/result.json"))
PARAMS = ("e", "h", "panel_dz", "panel_x0", "panel_y_end", "side_hf", "side_hb", "side_ye", "global_roll_deg", "global_dy", "halftop_dz", "halftop_xf", "halftop_ye", "edge_ext", "edge_lip")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def roi():
    with np.load(R / "visual_dev_bound_v1/dev3" / EP / "masks.npz") as a:
        m = a["rack"]
    ys, xs = np.nonzero(m)
    return max(0, ys.min() - PAD), min(256, ys.max() + 1 + PAD), max(0, xs.min() - PAD), min(256, xs.max() + 1 + PAD)


def window(s, key):
    p = json.loads((SETS[s][0] / key / "probe.json").read_text())
    a, b = p["dense_window"]
    return p, list(range(a, b + 1))


def sim(cand, out):
    from cowm.init.estimate_state import Scene
    from cowm.s6.plan_probe import apply_state
    y0, y1, x0, x1 = roi()
    res = json.loads(Path(cand).read_text())
    sc = Scene(res["world"])
    doc = json.loads((Path(res["states"]) / EP / "state.json").read_text())
    robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    arrs = {}
    for s, key in PROBES:
        probe, steps = window(s, key)
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
    ys = np.mgrid[:ms.shape[1], :ms.shape[2]][0]
    area = ms.sum((1, 2))
    return (ms * ys).sum((1, 2)) / np.maximum(area, 1), area


def stall(cy, area, steps):
    ent = np.nonzero(area > 2 * area[0])[0]
    if not len(ent):
        return None
    for i in range(ent[0], len(cy) - STALL_WIN):
        if np.abs(cy[i:i + STALL_WIN + 1] - cy[i]).max() <= STALL_TOL:
            return steps[i]
    return steps[-1]


def observed():
    obs, files = {}, {}
    for s, key in PROBES:
        _, steps = window(s, key)
        ms = []
        for t in steps:
            fr = SETS[s][1] / key / f"{EP}__f{t:03d}"
            m = json.loads((fr / "association.json").read_text())
            if m["masks_sha256"] != sha(fr / "masks.npz") or m["objects"]["robot"]["status"] != "matched":
                raise ValueError(f"invalid observed evidence {fr}")
            files[str(fr / "masks.npz")] = m["masks_sha256"]
            with np.load(fr / "masks.npz") as a:
                ms.append(a["robot"])
        obs[key] = np.stack(ms)
    return obs, files


def name(c):
    return f"{c.parent.parent.name}__{c.parent.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", nargs=2)
    a = ap.parse_args()
    if a.sim:
        return sim(*a.sim)
    (OUT / "sim").mkdir(parents=True, exist_ok=True)

    def run(c):
        dest = OUT / "sim" / f"{name(c)}.npz"
        if not dest.exists():
            with (OUT / "sim" / f"{name(c)}.log").open("w") as log:
                r = subprocess.run([sys.executable, __file__, "--sim", str(c), str(dest)],
                                   stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                raise RuntimeError((str(c), r.returncode))
    with ThreadPoolExecutor(12) as pool:
        list(pool.map(run, CANDS))
    print("SIM_DONE", len(CANDS), flush=True)
    for s, key in PROBES:
        n = len(json.loads((SETS[s][0] / key / "probe.json").read_text())["frames"])
        if len(list((SETS[s][1] / key).glob("*/association.json"))) < n:
            print("TRACKS_NOT_READY", key, flush=True)
            return
    y0, y1, x0, x1 = roi()
    obs, files = observed()
    obs = {k: v[:, y0:y1, x0:x1] for k, v in obs.items()}
    ref = {}
    for s, k in PROBES:
        _, steps = window(s, k)
        cy, area = centroid(obs[k])
        ref[k] = {"cy": cy, "stall": stall(cy, area, steps), "steps": steps}
    rows = []
    for c in CANDS:
        res = json.loads(c.read_text())
        with np.load(OUT / "sim" / f"{name(c)}.npz") as z:
            pred = {k: z[k] for _, k in PROBES}
        per = {}
        for s, k in PROBES:
            o, p = obs[k], pred[k]
            u = (o | p).sum((1, 2))
            iou = np.where(u > 0, (o & p).sum((1, 2)) / np.maximum(u, 1), 1.0)
            cy, area = centroid(p)
            st = stall(cy, area, ref[k]["steps"])
            per[k] = {"set": s, "roi_iou": round(float(iou.mean()), 4),
                      "settled_dcy_px": round(float(cy[-1] - ref[k]["cy"][-1]), 2),
                      "stall_sim": st, "stall_obs": ref[k]["stall"],
                      "stall_err": None if st is None or ref[k]["stall"] is None else abs(st - ref[k]["stall"])}
        row = {"cand": name(c), "params": {k: res[k] for k in PARAMS if k in res},
               "static_iou": round(res["static_iou"], 4), "per_probe": per,
               "roi_iou_mean": round(float(np.mean([v["roi_iou"] for v in per.values()])), 4),
               "roi_iou_rear16": round(float(np.mean([v["roi_iou"] for v in per.values() if v["set"] == "rear16"])), 4),
               "roi_iou_acc23": round(float(np.mean([v["roi_iou"] for v in per.values() if v["set"] == "acc23"])), 4),
               "roi_iou_acc25": round(float(np.mean([v["roi_iou"] for v in per.values() if v["set"] == "acc25"])), 4),
               "stall_err_acc25_neg_y": [per[k]["stall_err"] for k in ("acc24_x+060_y-040", "acc24_x+080_y-040",
                                                                      "acc24_x+100_y-040")],
               "roi_iou_acc27": round(float(np.mean([v["roi_iou"] for v in per.values() if v["set"] == "acc27"])), 4),
               "stall_err_acc27_neg_y": [per[k]["stall_err"] for k in ("acc26_x+020_y-040", "acc26_x+070_y-060",
                                                                      "acc26_x+090_y-060", "acc26_x+110_y-040")],
               "roi_iou_neg_y_all": round(float(np.mean([v["roi_iou"] for k, v in per.items() if "_y-" in k])), 4),
               "roi_iou_acc30": round(float(np.mean([v["roi_iou"] for v in per.values() if v["set"] == "acc30"])), 4),
               "roi_iou_ym060": round(float(np.mean([per[k]["roi_iou"] for k in YM060])), 4),
               "dcy_abs_max": round(max(abs(v["settled_dcy_px"]) for v in per.values()), 2),
               "stall_err_max": max((v["stall_err"] if v["stall_err"] is not None else 999) for v in per.values())}
        row["loss"] = round(1 - row["static_iou"] + 3 * (1 - row["roi_iou_mean"]), 4)
        ex = [v["roi_iou"] for k, v in per.items() if k not in YM060]
        row["loss_ex_ym060"] = round(1 - row["static_iou"] + 3 * (1 - float(np.mean(ex))), 4)
        rows.append(row)
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError(f"input changed {p}")
    rows.sort(key=lambda r: r["loss"])
    (OUT / "score.json").write_text(json.dumps({
        "GT_inputs": False, "Target_numbers_read": False, "script_sha256": sha(__file__),
        "roi": [int(y0), int(y1), int(x0), int(x1)], "observed_stall": {k: ref[k]["stall"] for _, k in PROBES},
        "n_observed_frames": len(files),
        "data_roles": "all 27 probes are fit evidence (v22/v24/v26/v28 Target results seen); acceptance needs a NEW unused probe set", "candidates": rows},
        indent=1))
    for r in rows:
        print(json.dumps({k: r[k] for k in ("cand", "params", "static_iou", "roi_iou_rear16", "roi_iou_acc23", "roi_iou_acc25", "roi_iou_acc27", "roi_iou_neg_y_all", "roi_iou_acc30", "roi_iou_ym060",
                                             "dcy_abs_max", "stall_err_max", "loss", "loss_ex_ym060")}), flush=True)
    print("DENSE_V30_DONE", flush=True)


if __name__ == "__main__":
    main()
