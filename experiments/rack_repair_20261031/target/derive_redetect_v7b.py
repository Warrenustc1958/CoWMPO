"""Derive robot track repair v7b from saved v7 outputs (no model run): keep ONLY re-detected components that
touch the image border (entering robot links); drop components kept merely for touching the track (they
inflate the mask along its rim in grasp videos). Components are those of (v7 - track) with the same
connectivity v7 used, so this equals running v7 with the track-adjacency condition removed. No GT.
usage: python derive_redetect_v7b.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
EP = "libero_goal_t0_i0"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def border_only(track, v7):
    add = v7 & ~track
    lab, n = label(add)
    b = np.zeros_like(add)
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    keep = np.zeros_like(add)
    for c in range(1, n + 1):
        comp = lab == c
        if (comp & b).any():
            keep |= comp
    return track | keep, int(keep.sum())


def rack():
    src = R / "rack_robot_redetect_v7"
    out = R / "rack_robot_redetect_v7b"
    out.mkdir(exist_ok=False)
    fr = json.loads((src / "freeze.json").read_text())
    for q in fr["queries"]:
        for s in q["steps"]:
            f = f"{EP}__f{s:03d}"
            with np.load(Path(q["track"]) / f / "masks.npz") as a:
                tr = a["robot"].astype(bool)
            with np.load(src / q["key"] / f / "masks.npz") as a:
                v7 = a["robot"].astype(bool)
            m, k = border_only(tr, v7)
            d = out / q["key"] / f
            d.mkdir(parents=True)
            np.savez_compressed(d / "masks.npz", robot=m)
            (d / "association.json").write_text(json.dumps({"episode": f, "algorithm": "robot_redetect_v7b",
                "objects": {"robot": {"status": "matched" if m.any() else "missing", "added_area": k}},
                "masks_sha256": sha(d / "masks.npz")}, indent=1))
    (out / "freeze.json").write_text(json.dumps(dict(fr, rule="v7b: border-touching components only",
                                                     derived_from=str(src)), indent=1))
    print("RACK_V7B_DONE")


def grasp(src_name, track_rel, out_name):
    src, track, out = R / src_name, R / track_rel, R / out_name
    out.mkdir(exist_ok=False)
    n = 0
    for d7 in sorted(src.glob("*__f*")):
        with np.load(track / d7.name / "masks.npz") as a:
            tr = a["robot"].astype(bool)
        with np.load(d7 / "masks.npz") as a:
            arrs = {k: a[k] for k in a.files}
        meta = json.loads((d7 / "association.json").read_text())
        m, k = border_only(tr, arrs["robot"].astype(bool))
        arrs["robot"] = m
        d = out / d7.name
        d.mkdir()
        np.savez_compressed(d / "masks.npz", **arrs)
        meta["objects"]["robot"] = dict(meta["objects"]["robot"], status="matched" if m.any() else "missing",
                                        redetect_added_area=k)
        meta["algorithm_robot"] = "robot_redetect_v7b"
        meta["masks_sha256"] = sha(d / "masks.npz")
        (d / "association.json").write_text(json.dumps(meta, indent=1))
        n += 1
    print("GRASP_V7B_DONE", out_name, n)


if __name__ == "__main__":
    rack()
    grasp("grasp_redetect_fit_v7", "grasp_video_fit_cpu_v2/frames", "grasp_redetect_fit_v7b")
    grasp("grasp_redetect_dev_v7", "grasp_video_dev_v1/grasp", "grasp_redetect_dev_v7b")
