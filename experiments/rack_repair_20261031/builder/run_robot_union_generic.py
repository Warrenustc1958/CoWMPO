"""Generalise the frozen robot parts+union repair (v4 anchors, v5 union rule) to more probe video sets.

Same episode (libero_goal_t0_i0) and the SAME frozen proximal/distal anchors as rack_robot_parts_cpu_v4; same
tracker script; union with the existing whole-robot track on every frame. No GT, no per-episode choice, no
change of the rule. Writes a new directory; existing outputs are read-only inputs.
usage: python run_robot_union_generic.py --prepare | --run
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

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
E = Path("/data_cf/cowmpo/evidence")
C = Path("/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v10")
ANCH = R / "rack_robot_parts_cpu_v4" / "anchors"
OUT = R / "rack_robot_union_generic_v6"
EP = "libero_goal_t0_i0"
SETS = {  # group: (plan(key) -> probe.json, public evidence dir, existing whole-robot track dir, keys)
    "multidir": (lambda k: R / "rack_multidir_preflight_dev_v1" / k / "probe.json", E / "codex_rack_multidir_dev_v1",
                 R / "rack_multidir_video_cpu_v1", ["xp_h220", "xm_h220", "ym_h100", "yp_h100"]),
    "touch": (lambda k: R / "rack_touch_dev_v1" / k / "probes" / EP / "probe.json", E / "codex_rack_touch_dev_v1",
              R / "rack_touch_video_v1", ["h070", "h170", "h270"]),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(exist_ok=False)
    recs, files = [], {}
    for p in sorted((ANCH / EP).iterdir()):
        files[str(p)] = sha(p)
    for group, (plan_of, ev, old, keys) in SETS.items():
        for key in keys:
            plan = plan_of(key)
            probe = json.loads(plan.read_text())
            public = ev / key
            q = json.loads((public / "query.json").read_text())
            if q.get("query_sha256") not in (None, sha(plan)) or q.get("frames", probe["frames"]) != probe["frames"]:
                raise ValueError(f"action evidence mismatch {group}/{key}")
            ready = OUT / "ready" / key
            for sub in ("initial", "probes"):
                (ready / sub / EP).mkdir(parents=True)
            shutil.copy2(public / f"{EP}__f000" / "frame0.png", ready / "initial" / EP / "frame0.png")
            shutil.copy2(plan, ready / "probes" / EP / "probe.json")
            for f in [plan, *[public / f"{EP}__f{s:03d}" / "frame0.png" for s in probe["frames"]]]:
                files[str(f)] = sha(f)
            for s in probe["frames"]:
                for n in ("association.json", "masks.npz"):
                    f = old / key / f"{EP}__f{s:03d}" / n
                    files[str(f)] = sha(f)
            recs.append({"group": group, "key": key, "plan": str(plan), "public": str(public), "old": str(old / key),
                         "steps": probe["frames"]})
    (OUT / "freeze.json").write_text(json.dumps({
        "queries": recs, "files": files, "script_sha256": sha(__file__),
        "tracker_sha256": sha(C / "track_probe_video_device.py"), "anchors": str(ANCH),
        "rule": "frozen v4 proximal/distal anchors of the same episode, independent part tracks, union with the "
                "existing whole-robot track (v5 rule); no GT, no per-key choice; generalisation run",
        "GT_inputs": False}, indent=1))
    print("UNION_GENERIC_FROZEN", len(recs), "queries", flush=True)


def run():
    fr = json.loads((OUT / "freeze.json").read_text())
    if fr["script_sha256"] != sha(__file__) or fr["tracker_sha256"] != sha(C / "track_probe_video_device.py"):
        raise ValueError("code changed since freeze")

    def one(rec):
        key = rec["key"]
        ready, parts, final = OUT / "ready" / key, OUT / "parts" / key, OUT / "frames" / key
        cmd = [sys.executable, str(C / "track_probe_video_device.py"), "--data-role", "dev", "--device", "cpu",
               "--cpu-threads", "8", "--episodes", EP, "--anchors", str(ANCH), "--initial-evidence",
               str(ready / "initial"), "--evidence", rec["public"], "--probes", str(ready / "probes"), "--out", str(parts)]
        with (OUT / f"{key}.log").open("w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError((key, r.returncode))
        exp = {f"{EP}__f{s:03d}" for s in rec["steps"]}
        if {q.parent.name for q in parts.glob("*/association.json")} != exp:
            raise ValueError(f"incomplete parts frames {key}")
        for fname in sorted(exp):
            union, areas = np.zeros((256, 256), bool), {}
            pm = json.loads((parts / fname / "association.json").read_text())
            if pm["masks_sha256"] != sha(parts / fname / "masks.npz"):
                raise ValueError("parts mask hash")
            with np.load(parts / fname / "masks.npz") as a:
                for n, o in pm["objects"].items():
                    areas[n] = int(a[n].sum())
                    if o["status"] == "matched":
                        union |= a[n]
            om = json.loads((Path(rec["old"]) / fname / "association.json").read_text())
            with np.load(Path(rec["old"]) / fname / "masks.npz") as a:
                whole = a["robot"]
            areas["whole"] = int(whole.sum())
            if om["objects"]["robot"]["status"] == "matched":
                union |= whole
            d = final / fname
            d.mkdir(parents=True)
            np.savez_compressed(d / "masks.npz", robot=union)
            (d / "association.json").write_text(json.dumps({
                "episode": fname, "algorithm": "robot_whole_parts_union_generic_v6",
                "objects": {"robot": {"status": "matched" if union.any() else "missing", "input_areas": areas,
                                      "union_area": int(union.sum())}},
                "masks_sha256": sha(d / "masks.npz")}, indent=1))
        print("UNION_QUERY_DONE", rec["group"], key, len(exp), flush=True)
        return {"group": rec["group"], "key": key, "frames": len(exp)}

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(one, fr["queries"]))
    for p, h in fr["files"].items():
        if sha(p) != h:
            raise ValueError(f"input changed {p}")
    (OUT / "complete.json").write_text(json.dumps(rows, indent=1))
    print("UNION_GENERIC_DONE", sum(r["frames"] for r in rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    prepare() if a.prepare else run()
