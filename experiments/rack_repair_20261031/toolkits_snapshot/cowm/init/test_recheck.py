"""CPU tests for cowm.init.recheck (synthetic masks, fake detector/segmenter, no models). Exit 1 on failure.

  coords            : crop around a box stays inside the image; detector boxes map back by the crop offset
  reject_alt        : ambiguous B (two regions); detector sees only another object's prompt on the alternative
                      -> alternative removed, B matched to the chosen region
  no_result         : detector returns nothing -> verdict no_evidence, pool and statuses unchanged
  budget            : 4 triggered objects -> 4th budget_exhausted, detector called <= 3 objects x 2 crops
  conflict_after    : a missing object's new candidate re-explains a matched object's pixels -> matched object
                      unchanged, the new candidate does not win
  missing_found     : a missing object's own-only detection in free space becomes a new candidate -> matched
"""
import sys

import numpy as np

from cowm.init import recheck as RC

H = W = 64


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def boxof(m):
    ys, xs = np.nonzero(m)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def cand(cid, obj, score, m):
    return {"cid": cid, "obj": obj, "score": score, "filtered": None, "box": boxof(m)}


def seg_rects(boxes):
    out = []
    for b in boxes:
        m = np.zeros((H, W), bool)
        m[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = True
        out.append(m)
    return out


def main():
    res = {}
    c = RC.crop_around([60, 60, 64, 64], W, H)
    res["coords"] = (0 <= c[0] < c[2] <= W and 0 <= c[1] < c[3] <= H
                     and RC.to_original([1, 2, 3, 4], [10, 20, 40, 50]) == [11.0, 22.0, 13.0, 24.0])
    # ambiguous B: chosen X (free space), alternative Y (another free spot, same score)
    ms = {0: rect(5, 25, 5, 25), 1: rect(40, 48, 2, 10), 2: rect(40, 48, 50, 58)}
    cs = [cand(0, "A", 0.7, ms[0]), cand(1, "B", 0.20, ms[1]), cand(2, "B", 0.20, ms[2])]
    base = RC.associate(["A", "B"], cs, ms)["objects"]
    alt = base["B"]["alt_cid"]

    def det_reject(crop):
        b = cs[alt]["box"]
        if box_in(b, crop):
            return [{"obj": "R", "prompt": "rack", "score": 0.3, "box": b}]
        return []
    out = RC.recheck_episode(["A", "B"], cs, ms, W, H, det_reject, seg_rects)
    fb = out["final"]["objects"]["B"]
    res["reject_alt"] = (base["B"]["status"] == "ambiguous" and alt in out["removed"]
                         and fb["status"] == "matched" and fb["cid"] == base["B"]["cid"])
    out = RC.recheck_episode(["A", "B"], cs, ms, W, H, lambda crop: [], seg_rects)
    res["no_result"] = (not out["removed"] and not out["added"] and out["final"]["objects"]["B"]["status"] == "ambiguous"
                        and all(e["verdict"] == "no_evidence" for e in out["log"]["B"]["regions"]))
    # budget: 4 objects without candidates -> missing
    calls = []
    out = RC.recheck_episode(["A", "C", "D", "E", "F"], cs[:1], ms, W, H,
                             lambda crop: calls.append(crop) or [], seg_rects)
    ex = [n for n, e in out["log"].items() if e.get("reason") == "budget_exhausted"]
    res["budget"] = ex == ["F"] and len(calls) <= 3 * 2 and out["model_calls"] == len(calls)
    # conflict after recheck: B missing; detector returns a B-only detection exactly on A's region
    def det_on_a(crop):
        b = cs[0]["box"]
        return [{"obj": "B", "prompt": "b", "score": 0.3, "box": b}] if box_in(b, crop) else []
    out = RC.recheck_episode(["A", "B"], cs[:1], ms, W, H, det_on_a, seg_rects)
    fo = out["final"]["objects"]
    res["conflict_after"] = (fo["A"]["status"] == "matched" and fo["A"]["cid"] == 0
                             and fo["B"]["status"] != "matched")
    # missing found in free space
    fb_box = boxof(rect(44, 52, 44, 52))
    def det_free(crop):
        return [{"obj": "B", "prompt": "b", "score": 0.3, "box": fb_box}] if box_in(fb_box, crop) else []
    out = RC.recheck_episode(["A", "B"], cs[:1], ms, W, H, det_free, seg_rects)
    fo = out["final"]["objects"]
    res["missing_found"] = (fo["B"]["status"] == "matched" and len(out["added"]) == 1
                            and fo["A"]["cid"] == 0 and not out["changed_untriggered"])
    ok = all(res.values())
    print(" ".join(f"{k}={v}" for k, v in res.items()))
    print("RECHECK_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def box_in(b, crop):
    return b[0] >= crop[0] and b[1] >= crop[1] and b[2] <= crop[2] and b[3] <= crop[3]


if __name__ == "__main__":
    main()
