# Copyright 2026 The RLinf Authors.
"""Rack anisotropic size scan v9 (width scaled separately from depth/height): are the rack's 3D dimensions identifiable from clean contact responses?

Single-view silhouette fitting trades size against depth (candidate01 fits the silhouette with loss 0.011 at
16.5 x 21.4 x 28.5 cm, far smaller than the target). Here the dims of the selected supported two-shelf model
are scaled jointly by SCALES; for every scale the per-episode rack pose (dx, dy, dyaw) is re-fitted to the
same six dev silhouettes with the dims FIXED, then the candidate is scored exactly like rack_clean_pose_v6
(static IoU + 3 x contact-phase robot-mask IoU) on the four verified gripper-only probes, using the frozen
v7b robot tracks. No Target data. Writes rack_scale_scan_dev_v8/.
usage: python fit_rack_scale_scan_v8.py [--index i]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.special import expit

from cowm import roles
from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree
from experimental_rack import build
from fit_rack_supported_silhouette import SIGNS, parts

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
BASE = R / "bowl_support_dev_v1/world11"
SRC = R / "rack_supported_silhouette_dev_v1/candidate01"
OUT = R / "rack_aniso_scan_dev_v9"
TRACK = R / "rack_robot_redetect_v7b"
QCFG = R / "rack_clean_pose_dev_v6/freeze.json"
EP = "libero_goal_t0_i0"
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]
SCALES = [(sw, sdh) for sw in (1.0, 1.15, 1.3, 1.45, 1.6) for sdh in (1.0, 1.12, 1.25)]  # (width, depth=height)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def queries():
    out = []
    for q in json.loads(QCFG.read_text())["queries"]:
        out.append(dict(q, video=str(TRACK / q["key"])))
    return out


def worker(i):
    roles.require(EPS, {"dev"}, "rack size scan")
    s = SCALES[i]
    fit0 = json.loads((SRC / "fit.json").read_text())
    dims = np.array(fit0["dims"]) * np.array([s[0], s[1], s[1]])
    incline = float(fit0["incline"])
    dest = OUT / f"scale{i:02d}"
    dest.mkdir()
    scene = Scene(str(BASE))
    scene.reset()
    m, d = scene.env.sim.model, scene.env.sim.data
    cid = m.camera_name2id("agentview")
    rot, camera = d.cam_xmat[cid].reshape(3, 3).copy(), d.cam_xpos[cid].copy()
    f = 128 / np.tan(np.radians(m.cam_fovy[cid]) / 2)
    top = scene.world["table"]["height"]
    scene.env.close()
    recs = []
    for ep in EPS:
        doc = json.loads((R / "stove_parts_states_dev_v1" / ep / "state.json").read_text())
        old = json.loads((SRC / "states" / ep / "state.json").read_text())["objects"]["rack"]
        with np.load(R / "visual_dev_bound_v1/dev3" / ep / "masks.npz") as a:
            mask = a["rack"].copy()
            other = np.logical_or.reduce([a[n] for n in a.files if n != "rack"])
        yy, xx = np.mgrid[0:256:3, 0:256:3]
        recs.append({"ep": ep, "doc": doc, "old": old, "points": np.stack([xx.ravel(), yy.ravel()], -1),
                     "target": gaussian_filter(mask.astype(float), .5)[yy, xx].ravel(),
                     "weight": (~other[yy, xx]).ravel().astype(float)})
    boxes = parts(dims[0], dims[1], dims[2], incline)

    def project(half, pos, pitch, xy, yaw):
        c, sn = np.cos(pitch), np.sin(pitch)
        rp = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
        c, sn = np.cos(yaw), np.sin(yaw)
        rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        pts = ((SIGNS * half) @ rp.T + pos) @ rz.T + np.array([*xy, top])
        p = (pts - camera) @ rot
        dep = -p[:, 2]
        uv = np.stack([255 - (128 + f * p[:, 0] / dep), 128 - f * p[:, 1] / dep], -1)
        return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(float)

    def residual(x):
        out = []
        for k, rec in enumerate(recs):
            dx, dy, dyaw = x[3 * k:3 * k + 3]
            xy = np.asarray(rec["old"]["pos"][:2]) + [dx, dy]
            yaw = rec["old"]["yaw"] + dyaw
            occ = np.zeros(len(rec["points"]))
            for half, pos, pitch in boxes:
                poly = project(half, pos, pitch, xy, yaw)
                edge = np.roll(poly, -1, axis=0) - poly
                delta = rec["points"][:, None, :] - poly[None, :, :]
                dist = (edge[None, :, 0] * delta[:, :, 1] - edge[None, :, 1] * delta[:, :, 0]) / \
                    np.maximum(np.linalg.norm(edge, axis=-1)[None, :], 1e-8)
                occ = np.maximum(occ, expit(dist.min(axis=1) / .5))
            out.extend((occ - rec["target"]) * rec["weight"])
        return np.asarray(out)

    def jac(x):
        cols = []
        for j in range(len(x)):
            dx = np.zeros_like(x)
            dx[j] = .0005
            cols.append((residual(x + dx) - residual(x - dx)) / (2 * dx[j]))
        return np.stack(cols, axis=1)

    x0 = np.zeros(3 * len(EPS))
    lo, hi = np.tile([-.20, -.20, -1.2], len(EPS)), np.tile([.20, .20, 1.2], len(EPS))
    fit = least_squares(residual, x0, jac=jac, bounds=(lo, hi), max_nfev=40, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    world = dest / "world"
    shutil.copytree(BASE, world)
    meta = json.loads((world / "world.json").read_text())
    ob = next(o for o in meta["objects"] if o["name"] == "rack")
    ob["attrs"].update(width=float(dims[0]), depth=float(dims[1]), height=float(dims[2]), incline=incline)
    ob["top_z"] = float(dims[2])
    ob["pos"][2] = top
    build("rack", world / "assets/rack", width=dims[0], depth=dims[1], height=dims[2], incline=incline)
    meta.pop("world_sha256")
    meta["rack_scale_scan"] = {"scale": s, "source": str(SRC), "pose": "re-fitted to six dev silhouettes, dims fixed"}
    meta["world_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    (world / "world.json").write_text(json.dumps(meta, indent=1))
    docs = {}
    for k, rec in enumerate(recs):
        dx, dy, dyaw = fit.x[3 * k:3 * k + 3]
        rec["doc"]["objects"]["rack"].update(pos=[rec["old"]["pos"][0] + dx, rec["old"]["pos"][1] + dy, top],
                                             yaw=float(rec["old"]["yaw"] + dyaw))
        docs[rec["ep"]] = rec["doc"]
        p = dest / "states" / rec["ep"]
        p.mkdir(parents=True)
        (p / "state.json").write_text(json.dumps(rec["doc"], indent=1))
    # score exactly like rack_clean_pose_v6
    wh = tree_sha(world)
    scene = Scene(str(world))
    static = []
    for ep in EPS:
        robot = json.loads((R / "robot_rgb_all_dev_v2" / ep / "robot.json").read_text())["robot"]
        scene.reset()
        apply_state(scene, docs[ep]["objects"], require_matched=True, robot=robot)
        kk = scene.names.index("rack")
        with np.load(R / "visual_dev_bound_v1/dev3" / ep / "masks.npz") as a:
            target = a["rack"]
        pred = scene.seg() == kk
        static.append(iou(pred, target))
        np.savez_compressed(dest / "states" / ep / "render_masks.npz", rack=pred)
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
            meta_f = json.loads((fr / "association.json").read_text())
            if meta_f["masks_sha256"] != sha(fr / "masks.npz") or meta_f["objects"]["robot"]["status"] != "matched":
                raise ValueError("invalid response evidence")
            with np.load(fr / "masks.npz") as a:
                vals.append(iou(scene.robot_mask(), a["robot"]))
        if len(vals) != len(q["frames"]):
            raise ValueError("incomplete contact response")
        responses.append({"key": q["key"], "mean_iou": float(np.mean(vals)), "ious": vals})
    scene.env.close()
    verify_tree(world, wh)
    rep = {"index": i, "scale": s, "dims_m": dims.tolist(), "incline": incline, "silhouette_mse": float(np.mean(fit.fun ** 2)),
           "pose_offsets": fit.x.reshape(-1, 3).tolist(), "world": str(world), "world_tree_sha256": wh,
           "states": str(dest / "states"), "static_iou": float(np.mean(static)),
           "robot_iou": float(np.mean([r["mean_iou"] for r in responses])), "responses": responses}
    rep["loss"] = 1 - rep["static_iou"] + 3 * (1 - rep["robot_iou"])
    (dest / "result.json").write_text(json.dumps(rep, indent=1))
    print("SCALE_DONE", s, round(rep["silhouette_mse"], 5), round(rep["static_iou"], 4), round(rep["robot_iou"], 4),
          round(rep["loss"], 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.index is not None:
        return worker(a.index)
    OUT.mkdir(exist_ok=False)
    qs = queries()
    files = {str(SRC / "fit.json"): sha(SRC / "fit.json")}
    for q in qs:
        files[q["source"]] = sha(q["source"])
        for st in q["frames"]:
            for n in ("masks.npz", "association.json"):
                p = Path(q["video"]) / f"{EP}__f{st:03d}" / n
                files[str(p)] = sha(p)
    (OUT / "freeze.json").write_text(json.dumps({
        "queries": qs, "scales": SCALES, "files": files, "base_world_tree_sha256": tree_sha(BASE),
        "script_sha256": sha(__file__), "GT_inputs": False,
        "objective": "1 - static IoU + 3 x (1 - contact-phase robot IoU); four verified gripper-only probes; v7b tracks",
        "question": "does the contact response discriminate rack size, given pose re-fitted per size?"}, indent=1))

    def run(i):
        with (OUT / f"scale{i:02d}.log").open("w") as log:
            r = subprocess.run([sys.executable, __file__, "--index", str(i)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((i, r.returncode))
        return json.loads((OUT / f"scale{i:02d}" / "result.json").read_text())

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(run, range(len(SCALES))))
    for p, h in json.loads((OUT / "freeze.json").read_text())["files"].items():
        if sha(p) != h:
            raise ValueError(f"input changed {p}")
    brief = [{k: v for k, v in r.items() if k not in ("responses", "pose_offsets")} for r in rows]
    best = min(brief, key=lambda r: r["loss"])
    (OUT / "selection.json").write_text(json.dumps({"best": best, "candidates": brief}, indent=1))
    print("SCALE_SCAN_SELECTED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
