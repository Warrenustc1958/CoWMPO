"""S5 identity/localisation diagnostic (Target side ONLY; reads GT, output never shown to the Builder/Agent).

Per episode and GT object, three nested outcomes (audit 10.29: "observed" overstated perception):
  has_mask    : perception returned a non-rejected mask for the object
  identity_ok : has_mask AND the mask's best-IoU GT object is this object AND that IoU >= ID_IOU
                (otherwise the object it actually covers is recorded: wrong association)
  loc_ok      : identity_ok AND AABB-centre xy error < LOC_CM (same error definition as s5_compare)
Also: for wrong associations, the state estimate's fitted IoU to the WRONG mask (high = the estimator
fitted the wrong object well).
usage: python s5_identity_diag.py --est <estimate_state root> --gt <gt_dump dir> --out <report.json>
"""
import argparse
import json
import os

import numpy as np

ID_IOU = 0.3
LOC_CM = 3.0


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--est", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--masks", default="", help="perception masks root (<ep>/masks.npz); default: --est")
    a = p.parse_args()
    mroot = a.masks or a.est
    eps = sorted(e for e in os.listdir(a.est) if os.path.isfile(os.path.join(a.est, e, "state.json")))
    per_obj, wrong = {}, []
    for ep in eps:
        st = json.load(open(os.path.join(a.est, ep, "state.json")))["objects"]
        gt = json.load(open(os.path.join(a.gt, ep + ".json")))
        gm = np.load(os.path.join(a.gt, ep + "_masks.npz"))
        sm = np.load(os.path.join(mroot, ep, "masks.npz"))
        for n, g in gt.items():
            r = per_obj.setdefault(n, {"n": 0, "has_mask": 0, "identity_ok": 0, "loc_ok": 0, "err_cm_id_ok": []})
            r["n"] += 1
            s = st.get(n, {})
            if n not in sm.files or not sm[n].any() or s.get("mask_rejected"):
                continue
            r["has_mask"] += 1
            ious = {k: iou(sm[n], gm[k]) for k in gm.files}
            best = max(ious, key=ious.get)
            if best != n or ious[n] < ID_IOU:
                wrong.append({"episode": ep, "object": n, "iou_own_gt": round(ious[n], 3),
                              "covers": best, "iou_covered": round(ious[best], 3),
                              "fitted_iou_to_mask": s.get("iou")})
                continue
            r["identity_ok"] += 1
            if s.get("aabb_center_rel_base") and g.get("aabb_center_rel_base"):
                e = np.array(s["aabb_center_rel_base"][:2]) - np.array(g["aabb_center_rel_base"][:2])
            else:
                e = np.array(s["pos_rel_base"][:2]) - np.array(g["pos_rel_base"][:2])
            err = float(np.linalg.norm(e) * 100)
            r["err_cm_id_ok"].append(err)
            r["loc_ok"] += int(err < LOC_CM)
    summ = {}
    for n, r in per_obj.items():
        e = r.pop("err_cm_id_ok")
        summ[n] = dict(r, err_cm_identity_ok_mean=round(float(np.mean(e)), 2) if e else None,
                       err_cm_identity_ok_p90=round(float(np.quantile(e, 0.9)), 2) if e else None)
    tot = {k: sum(r[k] for r in per_obj.values()) for k in ("n", "has_mask", "identity_ok", "loc_ok")}
    rep = {"est": os.path.abspath(a.est), "gt": os.path.abspath(a.gt), "episodes": len(eps),
           "thresholds": {"identity_iou": ID_IOU, "loc_cm": LOC_CM}, "total": tot,
           "rates": {k: round(tot[k] / tot["n"], 3) for k in ("has_mask", "identity_ok", "loc_ok")},
           "per_object": summ, "wrong_associations": wrong}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print("S5_ID_DIAG", json.dumps({"episodes": len(eps), "total": tot, "rates": rep["rates"],
                                    "n_wrong": len(wrong)}))
    for n, r in summ.items():
        print(f"  {n:13s} n={r['n']:2d} mask={r['has_mask']:2d} id_ok={r['identity_ok']:2d} loc_ok={r['loc_ok']:2d} "
              f"err_mean={r['err_cm_identity_ok_mean']}")
    for w in wrong:
        print("  WRONG", w)


if __name__ == "__main__":
    main()
