"""Merge step of the generalised robot union (v6), fixed: only MATCHED parts are read (missing parts are not
stored in masks.npz, as in v4). Reads the completed parts tracks and the frozen whole-robot tracks; writes a
fresh frames_merged/ (the partial frames/ from the failed merge is left untouched)."""
import hashlib
import json
from pathlib import Path

import numpy as np

O = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020/rack_robot_union_generic_v6")
EP = "libero_goal_t0_i0"
OUT = O / "frames_merged"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


fr = json.loads((O / "freeze.json").read_text())
for p, h in fr["files"].items():
    if sha(p) != h:
        raise ValueError(f"input changed {p}")
OUT.mkdir(exist_ok=False)
rows = []
for rec in fr["queries"]:
    parts, old = O / "parts" / rec["key"], Path(rec["old"])
    n = 0
    for s in rec["steps"]:
        f = f"{EP}__f{s:03d}"
        pm = json.loads((parts / f / "association.json").read_text())
        if pm["masks_sha256"] != sha(parts / f / "masks.npz"):
            raise ValueError("parts mask hash")
        union, areas = np.zeros((256, 256), bool), {}
        with np.load(parts / f / "masks.npz") as a:
            for name, o in pm["objects"].items():
                if o["status"] == "matched":
                    union |= a[name]
                    areas[name] = int(a[name].sum())
                else:
                    areas[name] = None
        om = json.loads((old / f / "association.json").read_text())
        with np.load(old / f / "masks.npz") as a:
            whole = a["robot"]
        areas["whole"] = int(whole.sum())
        if om["objects"]["robot"]["status"] == "matched":
            union |= whole
        d = OUT / rec["key"] / f
        d.mkdir(parents=True)
        np.savez_compressed(d / "masks.npz", robot=union)
        (d / "association.json").write_text(json.dumps({
            "episode": f, "algorithm": "robot_whole_parts_union_generic_v6",
            "objects": {"robot": {"status": "matched" if union.any() else "missing", "input_areas": areas,
                                  "union_area": int(union.sum())}},
            "masks_sha256": sha(d / "masks.npz")}, indent=1))
        n += 1
    rows.append({"group": rec["group"], "key": rec["key"], "frames": n})
(O / "complete.json").write_text(json.dumps(rows, indent=1))
print("UNION_MERGE_DONE", sum(r["frames"] for r in rows))
