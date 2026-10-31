"""Content check of an rm_natural_control_v2 run directory (called by run_rm_natural_v2.sh). Exit 1 on failure.

  identity : report.json episodes {(task_id, trial)} == requested tasks x trials, no duplicates
  files    : exactly one episodes/t<task>_i<trial>.npz per requested pair, nothing else
  content  : every npz loads and has actions [T,7], labels [T+1], p [T+1], frames [K,256,256,3] uint8,
             frame_idx [K] (starts at 0, increasing, <= T), task_id/trial matching its file name;
             p finite in [0,1]; report first_success consistent with labels
usage: python validate_rm_run.py <out_dir> <tasks csv> <trials csv>
"""
import json
import os
import sys

import numpy as np


def main():
    out, tasks, trials = sys.argv[1], sys.argv[2], sys.argv[3]
    want = {(int(t), int(i)) for t in tasks.split(",") for i in trials.split(",")}
    errs = []
    rep = json.load(open(os.path.join(out, "report.json")))
    got = [(int(e["task_id"]), int(e["trial"])) for e in rep.get("episodes", [])]
    if len(got) != len(set(got)):
        errs.append("duplicate episodes in report")
    if set(got) != want:
        errs.append(f"report episodes {sorted(set(got))} != requested {sorted(want)}")
    ed = os.path.join(out, "episodes")
    files = sorted(os.listdir(ed)) if os.path.isdir(ed) else []
    expect_files = sorted(f"t{t}_i{i}.npz" for t, i in want)
    if files != expect_files:
        errs.append(f"episode files {files} != expected {expect_files}")
    by_ep = {(int(e["task_id"]), int(e["trial"])): e for e in rep.get("episodes", [])}
    for t, i in sorted(want):
        f = os.path.join(ed, f"t{t}_i{i}.npz")
        if not os.path.isfile(f):
            continue
        try:
            z = np.load(f)
            a, lab, p, fr, fi = z["actions"], z["labels"], z["p"], z["frames"], z["frame_idx"]
            tid, tri = int(z["task_id"]), int(z["trial"])
        except Exception as ex:  # noqa: BLE001
            errs.append(f"{f}: cannot load ({type(ex).__name__}: {ex})")
            continue
        T = a.shape[0]
        bad = []
        if a.ndim != 2 or a.shape[1] != 7 or T == 0:
            bad.append(f"actions {a.shape}")
        if lab.shape != (T + 1,) or p.shape != (T + 1,):
            bad.append(f"labels {lab.shape} p {p.shape} vs T+1={T + 1}")
        if fr.ndim != 4 or fr.shape[1:] != (256, 256, 3) or fr.dtype != np.uint8 or fr.shape[0] != fi.shape[0]:
            bad.append(f"frames {fr.shape} {fr.dtype} frame_idx {fi.shape}")
        if fi.size and (fi[0] != 0 or np.any(np.diff(fi) <= 0) or fi[-1] > T):
            bad.append("frame_idx not 0-based increasing within T")
        if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            bad.append("p not finite in [0,1]")
        if (tid, tri) != (t, i):
            bad.append(f"content says t{tid}_i{tri}")
        e = by_ep.get((t, i), {})
        fs = e.get("first_success")
        lab_fs = int(np.argmax(lab[1:])) + 1 if lab[1:].any() else None   # first success after an action
        if "first_success" in e and fs != lab_fs:
            bad.append(f"report first_success {fs} != labels {lab_fs}")
        if bad:
            errs.append(f"{f}: " + "; ".join(bad))
    for e in errs:
        print("INVALID", e)
    print("RUN_CONTENT", "OK" if not errs else "INVALID", len(want), "episodes")
    sys.exit(0 if not errs else 1)


if __name__ == "__main__":
    main()
