"""S5 budgeted local recheck of ambiguous / missing objects (Builder side; no GT; pure logic, models injected).

Rules fixed before the dev run (2026-10-30):
  trigger   : objects that associate() marks ambiguous or missing; order = ambiguous by margin ascending, then
              missing by name. At most MAX_OBJECTS per episode, MAX_CROPS crops per object, no recursion;
              objects past the budget keep their status with reason "budget_exhausted".
  detection : every crop runs the SAME prompt protocol (all prompts of all world objects), threshold fixed
              in the injected detector. Detections are returned in ORIGINAL image coordinates.
  verdict   : a detection supports a region if box IoU(detection, region box) >= REGION_BOX_IOU. Presence only,
              no score comparison across prompts or crops (scores are not calibrated):
                own prompt only -> confirmed | other prompts only -> rejected | both -> contested | none ->
                no_evidence
  effect    : only "rejected" changes the pool: the region's candidate and same-region duplicates of that
              object (containment >= SAME_REGION_FRAC) are removed. confirmed / contested / no_evidence leave
              the pool unchanged.
  ambiguous : check the chosen region and the alternative region (crop = box + CTX of its size on each side).
  missing   : tiles of TILE px (stride TILE/2) ranked by the fraction not covered by matched objects' masks
              (priority only, nothing excluded); a new candidate = own-prompt detection that no other prompt
              supports, passing the prior filter; masks from the injected segmenter on the full image.
  then      : associate() is re-run on the updated pool (status, margin and other objects recomputed).
Known limits: recheck detection scores (used as unary for new candidates of missing objects) are not
calibrated against first-pass scores; upscaling a crop changes detection scale but adds no new detail.
"""
import numpy as np

from cowm.init.associate import SAME_REGION_FRAC, associate, box_iou, mask_overlap

CTX = 0.5
MIN_CROP = 64
REGION_BOX_IOU = 0.5
MAX_OBJECTS = 3
MAX_CROPS = 2
TILE = 128


def crop_around(box, W, H, ctx=CTX, min_size=MIN_CROP):
    x0, y0, x1, y1 = [float(v) for v in box]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max((x1 - x0) * (1 + 2 * ctx), min_size)
    h = max((y1 - y0) * (1 + 2 * ctx), min_size)
    return [int(max(0, np.floor(cx - w / 2))), int(max(0, np.floor(cy - h / 2))),
            int(min(W, np.ceil(cx + w / 2))), int(min(H, np.ceil(cy + h / 2)))]


def to_original(box_in_crop, crop):
    """Detector boxes are in crop pixels (post-processed at the crop's own size): add the crop offset."""
    return [float(box_in_crop[0]) + crop[0], float(box_in_crop[1]) + crop[1],
            float(box_in_crop[2]) + crop[0], float(box_in_crop[3]) + crop[1]]


def tiles(W, H, tile=TILE):
    step = tile // 2
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    if xs[-1] != max(W - tile, 0):
        xs.append(max(W - tile, 0))
    if ys[-1] != max(H - tile, 0):
        ys.append(max(H - tile, 0))
    return [[x, y, min(x + tile, W), min(y + tile, H)] for y in ys for x in xs]


def region_verdict(obj, region_box, dets):
    hit = sorted({d["obj"] for d in dets if box_iou(d["box"], region_box) >= REGION_BOX_IOU})
    own, others = obj in hit, [o for o in hit if o != obj]
    v = ("contested" if own and others else "confirmed" if own else "rejected" if others else "no_evidence")
    return v, hit


def recheck_episode(names, candidates, masks, W, H, detect, segment, prior_filter=None, assoc_kw=None,
                    max_objects=MAX_OBJECTS, max_crops=MAX_CROPS):
    """detect(crop_box) -> [{obj, prompt, score, box(original px)}] for the full prompt protocol;
    segment([boxes]) -> [bool HxW]; prior_filter(box, obj) -> reason or None.
    Returns base / final association, per-object log, pool changes and model-call counts."""
    kw = dict(assoc_kw or {})
    base = associate(names, candidates, masks, **kw)
    bo = base["objects"]
    trig = sorted([(0, bo[n]["margin"], n) for n in bo if bo[n]["status"] == "ambiguous"]) + \
        sorted([(1, 0.0, n) for n in bo if bo[n]["status"] == "missing"])
    by_cid = {c["cid"]: c for c in candidates}
    masks = dict(masks)
    removed, added, log, calls = set(), [], {}, 0
    next_cid = max(by_cid) + 1 if by_cid else 0
    for k, (kind, _, n) in enumerate(trig):
        if k >= max_objects:
            log[n] = {"action": "none", "reason": "budget_exhausted", "status_before": bo[n]["status"]}
            continue
        r = bo[n]
        if kind == 0:
            regions = [("chosen", r["cid"])] + ([("alt", r["alt_cid"])] if r["alt_cid"] is not None else [])
            entries = []
            for role, cid in regions[:max_crops]:
                c = by_cid[cid]
                crop = crop_around(c["box"], W, H)
                dets = detect(crop)
                calls += 1
                v, hit = region_verdict(n, c["box"], dets)
                rm = []
                if v == "rejected":
                    rm = [cj["cid"] for cj in candidates if cj["obj"] == n and cj["cid"] not in removed
                          and mask_overlap(masks[cj["cid"]], masks[cid]) >= SAME_REGION_FRAC]
                    removed.update(rm)
                entries.append({"role": role, "cid": cid, "crop": crop, "verdict": v, "supported_by": hit,
                                "n_dets": len(dets), "removed": rm})
            log[n] = {"action": "ambiguous_recheck", "status_before": "ambiguous", "regions": entries}
        else:
            occ = np.zeros((H, W), bool)
            for m, rm_ in bo.items():
                if rm_["status"] == "matched":
                    occ |= masks[rm_["cid"]]
            ts = sorted(tiles(W, H), key=lambda t: (-(1.0 - occ[t[1]:t[3], t[0]:t[2]].mean()), t))[:max_crops]
            found, entries = [], []
            for t in ts:
                dets = detect(t)
                calls += 1
                kept = []
                for d in dets:
                    if d["obj"] != n:
                        continue
                    v, hit = region_verdict(n, d["box"], dets)
                    why = prior_filter(d["box"], n) if prior_filter else None
                    if v == "confirmed" and why is None:
                        kept.append(d)
                entries.append({"tile": t, "free_frac": round(float(1.0 - occ[t[1]:t[3], t[0]:t[2]].mean()), 3),
                                "n_dets": len(dets), "own_confirmed": len(kept)})
                found += kept
            uniq = []
            for d in sorted(found, key=lambda d: (-float(d["score"]), tuple(d["box"]))):
                if all(box_iou(d["box"], u["box"]) < 0.7 for u in uniq):
                    uniq.append(d)
            new_cids = []
            if uniq:
                ms = segment([u["box"] for u in uniq])
                for u, mk in zip(uniq, ms):
                    c = {"cid": next_cid, "obj": n, "prompt": u["prompt"], "score": float(u["score"]),
                         "box": [float(v) for v in u["box"]], "filtered": None, "source": "recheck"}
                    masks[next_cid] = np.asarray(mk, bool)
                    added.append(c)
                    new_cids.append(next_cid)
                    next_cid += 1
            log[n] = {"action": "missing_search", "status_before": "missing", "tiles": entries,
                      "new_candidates": new_cids}
    pool = [c for c in candidates if c["cid"] not in removed] + added
    final = associate(names, pool, masks, **kw)
    for n, e in log.items():
        e["status_after"] = final["objects"][n]["status"]
        e["cid_after"] = final["objects"][n]["cid"]
    changed_untriggered = sorted(n for n in names if n not in log and
                                 (final["objects"][n]["status"], final["objects"][n]["cid"]) !=
                                 (bo[n]["status"], bo[n]["cid"]))
    return {"base": base, "final": final, "log": log, "removed": sorted(removed), "added": added,
            "model_calls": calls, "changed_untriggered": changed_untriggered, "masks": masks}
