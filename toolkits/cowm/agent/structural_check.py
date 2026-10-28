"""CoWM-PO structural consistency check (Builder side; feeds the Agent's structural-repair loop).

A world can pass every verifier gate and still be structurally wrong: an object declared with the wrong
template (a plate built as a box), or an evidence object missing from the program. Parameter fitting cannot
fix either. This check compiles the program once, pose-estimates it on a few evidence episodes and compares
the world's render masks with the evidence masks object by object:
  - missing     : an evidence mask has no object of that name in the world
  - unobserved  : a world object has no evidence mask (extra / misnamed object)
  - mismatch    : median over episodes of |fill(evidence) - fill(render)| exceeds FILL_TOL (wrong template
                  shape class); a low IoU with matching fill is only a non-blocking 'size' note
  - unlocalized : IoU below LOC_MIN (pose estimation failed; silhouettes are not comparable)
Only Builder-visible inputs are read (program, evidence masks); target GT is never touched.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from cowm.world.compile_world import compile_world

EVIDENCE = "/data_cf/cowmpo/evidence/libero_goal_dev3"
FILL_TOL = 0.08   # max |fill(evidence) - fill(render)| for a template of the right shape class
LOC_MIN = 0.15    # below this IoU the pose estimate failed and the fill comparison is meaningless


def silhouette(mask):
    """Fill ratio of the mask inside its bbox (disc ~0.79, box ~1.0) and bbox aspect (w/h)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    return {"fill": float(mask.sum() / (w * h)), "aspect": float(w / h), "area": int(mask.sum())}


def mean_stats(stats):
    stats = [s for s in stats if s]
    if not stats:
        return None
    return {k: round(float(np.mean([s[k] for s in stats])), 3) for k in stats[0]}


def check(program, masks, eps, work, iou_min, values=None):
    wdir, sdir = os.path.join(work, "world"), os.path.join(work, "state")
    compile_world(program, wdir, values)
    world_objs = {o["name"]: o for o in json.load(open(os.path.join(wdir, "world.json")))["objects"]}
    subprocess.run([sys.executable, "-m", "cowm.init.estimate_state", "--world", wdir, "--evidence", EVIDENCE,
                    "--masks", masks, "--out", sdir, "--episodes", ",".join(eps)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ev_names, iou, ev_sil, rd_sil, gaps = set(), {}, {}, {}, {}
    for ep in eps:
        ev = np.load(os.path.join(masks, ep, "masks.npz"))
        rd = np.load(os.path.join(sdir, ep, "render_masks.npz"))
        st = json.load(open(os.path.join(sdir, ep, "state.json")))["objects"]
        for n in ev.files:
            ev_names.add(n)
            es = silhouette(ev[n])
            ev_sil.setdefault(n, []).append(es)
            if n in rd.files:
                rs = silhouette(rd[n])
                rd_sil.setdefault(n, []).append(rs)
                if es and rs:
                    gaps.setdefault(n, []).append(abs(es["fill"] - rs["fill"]))
            if st.get(n, {}).get("iou") is not None:
                iou.setdefault(n, []).append(st[n]["iou"])
    findings = []
    for n in sorted(ev_names - set(world_objs)):
        findings.append({"kind": "missing", "object": n, "evidence": mean_stats(ev_sil[n]),
                         "hint": f"the evidence shows an object '{n}' that the program does not declare"})
    for n in sorted(set(world_objs) - ev_names):
        findings.append({"kind": "unobserved", "object": n, "category": world_objs[n].get("category"),
                         "hint": f"'{n}' has no counterpart in the evidence (extra or misnamed object)"})
    per_obj, notes = {}, []
    for n in sorted(ev_names & set(world_objs)):
        m = float(np.mean(iou[n])) if iou.get(n) else 0.0
        e, r = mean_stats(ev_sil[n]), mean_stats(rd_sil.get(n, []))
        # median over episodes: one episode stuck in a bad pose optimum (occluded render) must not flip the verdict
        gap = float(np.median(gaps[n])) if gaps.get(n) else None
        per_obj[n] = {"iou": round(m, 3), "category": world_objs[n].get("category"), "evidence": e, "render": r,
                      "fill_gap_median": None if gap is None else round(gap, 3)}
        # fill ratio is size-invariant: correct templates match within ~0.03 even at unfitted sizes (IoU 0.29),
        # so a fill gap means the shape class is wrong; low IoU with matching fill is left to fit_shape
        if m < LOC_MIN:   # pose estimation did not lock on: silhouettes are not comparable
            findings.append({"kind": "unlocalized", "object": n, **per_obj[n],
                             "hint": f"the world's '{n}' could not be aligned with the evidence (IoU {m:.2f}); "
                                     f"check its nominal placement and template"})
        elif gap is not None and gap > FILL_TOL:
            findings.append({"kind": "mismatch", "object": n, **per_obj[n],
                             "hint": f"silhouette fill ratio differs (evidence {e['fill']}, render {r['fill']}): "
                                     f"the template '{per_obj[n]['category']}' likely has the wrong shape class, "
                                     f"not just the wrong size"})
        elif m < iou_min:
            notes.append({"kind": "size", "object": n, "iou": per_obj[n]["iou"],
                          "hint": "silhouette class matches; size/placement is off (shape fitting handles this)"})
    return {"ok": not findings, "iou_min": iou_min, "fill_tol": FILL_TOL, "episodes": eps, "objects": per_obj,
            "findings": findings, "notes": notes}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--program", required=True)
    p.add_argument("--masks", required=True, help="EvidenceAdapter output (<episode>/masks.npz)")
    p.add_argument("--episodes", required=True, help="comma list of evidence episodes")
    p.add_argument("--out", required=True)
    p.add_argument("--iou-min", type=float, default=0.5)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = check(a.program, a.masks, a.episodes.split(","), os.path.join(a.out, "work"), a.iou_min)
    with open(os.path.join(a.out, "structural_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("STRUCTURAL", "OK" if rep["ok"] else "FINDINGS",
          json.dumps({n: o["iou"] for n, o in rep["objects"].items()}),
          json.dumps([(f["kind"], f["object"]) for f in rep["findings"]]), flush=True)


if __name__ == "__main__":
    main()
