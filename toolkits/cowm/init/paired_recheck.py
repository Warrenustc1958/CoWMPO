# Copyright 2026 The RLinf Authors.
"""Dev-only shared-context evidence collection; it never promotes an uncertain identity."""
import numpy as np

from cowm.init.associate import associate, box_iou
from cowm.init.recheck import MAX_OBJECTS, MAX_CROPS, REGION_BOX_IOU, crop_around, tiles


def paired_scores(regions, detections):
    """Compare the same exact prompt within one detector input; absent hits remain censored.

    These are ranking observations, not calibrated probabilities or acceptance decisions.
    """
    prompts = sorted({(d["obj"], d["prompt"]) for d in detections})
    rows = []
    for obj, prompt in prompts:
        scores = {}
        for role, box in regions.items():
            hits = [d for d in detections if d["obj"] == obj and d["prompt"] == prompt and
                    box_iou(d["box"], box) >= REGION_BOX_IOU]
            scores[role] = max((float(d["score"]) for d in hits), default=None)
        a, b = scores.get("chosen"), scores.get("alt")
        preference = "unresolved"
        if a is not None and b is not None:
            preference = "chosen" if a > b else "alt" if b > a else "tie"
        rows.append({"obj": obj, "prompt": prompt, "scores": scores, "preference": preference})
    return rows


def audit_episode(names, candidates, masks, W, H, detect, max_objects=MAX_OBJECTS, max_crops=MAX_CROPS):
    """Collect both region scores from one common crop and preserve the v3 association verbatim."""
    base = associate(names, candidates, masks)
    objects = base["objects"]
    by_cid = {c["cid"]: c for c in candidates}
    triggers = sorted((0, objects[n]["margin"], n) for n in names if objects[n]["status"] == "ambiguous")
    triggers += sorted((1, 0., n) for n in names if objects[n]["status"] == "missing")
    log, calls = {}, 0
    for index, (kind, _, name) in enumerate(triggers):
        rec = objects[name]
        entry = {"status_before": rec["status"], "status_after": rec["status"], "cid_after": rec["cid"],
                 "action": "audit_only"}
        log[name] = entry
        if index >= max_objects or max_crops < 1:
            entry["reason"] = "budget_exhausted"
            continue
        if kind == 0:
            if rec["alt_cid"] is None:
                entry["reason"] = "alternative_is_unmatched"
                continue
            regions = {role: by_cid[cid]["box"] for role, cid in
                       (("chosen", rec["cid"]), ("alt", rec["alt_cid"]))}
            boxes = list(regions.values())
            union = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes)]
            crop = crop_around(union, W, H)
            dets = detect(crop)
            calls += 1
            entry.update(crop=crop, regions=regions, paired_scores=paired_scores(regions, dets))
        else:
            occupied = np.zeros((H, W), bool)
            for r in objects.values():
                if r["status"] == "matched":
                    occupied |= masks[r["cid"]]
            crops = sorted(tiles(W, H), key=lambda t: (
                -float(1 - occupied[t[1]:t[3], t[0]:t[2]].mean()), t))[:max_crops]
            for crop in crops:
                detect(crop)
                calls += 1
            entry.update(crops=crops, reason="raw_candidates_only_no_identity_confirmation")
    return {"base": base, "final": base, "log": log, "removed": [], "added": [], "masks": masks,
            "model_calls": calls, "changed_untriggered": []}
