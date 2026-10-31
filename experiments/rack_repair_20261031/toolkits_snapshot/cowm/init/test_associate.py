"""CPU unit tests for cowm.init.associate (synthetic masks, no models, no GT). Exit code 1 on any failure.

  duplicate_claim   : the higher-scoring candidate of B lies inside A's mask; conflict mode picks B's other
                      candidate, score-only mode keeps the duplicate (shows the conflict term is what fixes it)
  legal_occlusion   : B's box lies inside A's box but the visible masks are disjoint -> both matched, no penalty
  all_bad           : B's only candidate re-explains A's pixels -> B missing, A matched
  ambiguous         : B has two equally scored conflict-free candidates in different places -> ambiguous
  order_invariance  : shuffling candidate order, object order and cids' list order gives identical output
  no_candidates     : an object without any candidate is missing, the rest unaffected
  duplicate_region_not_ambiguous : two near-identical candidates of one region -> matched, not ambiguous
  part_whole_not_ambiguous       : a candidate contained in another of the same object -> matched
"""
import random
import sys

import numpy as np

from cowm.init.associate import associate

H = W = 64


def rect(y0, y1, x0, x1, hole=None):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    if hole:
        m[hole[0]:hole[1], hole[2]:hole[3]] = False
    return m


def cand(cid, obj, score, m):
    ys, xs = np.nonzero(m)
    return {"cid": cid, "obj": obj, "score": score, "filtered": None,
            "box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]}


def run(cs, ms, names, **kw):
    return associate(names, cs, ms, **kw)["objects"]


def main():
    res = {}
    # duplicate claim
    ms = {0: rect(10, 30, 10, 30), 1: rect(14, 22, 14, 22), 2: rect(40, 48, 40, 48)}
    cs = [cand(0, "A", 0.7, ms[0]), cand(1, "B", 0.15, ms[1]), cand(2, "B", 0.10, ms[2])]
    r = run(cs, ms, ["A", "B"])
    rs = run(cs, ms, ["A", "B"], mode="score_only", abstain=False)
    res["duplicate_claim"] = (r["A"]["cid"] == 0 and r["B"]["cid"] == 2 and r["B"]["status"] == "matched"
                              and rs["B"]["cid"] == 1)
    # legal occlusion: B sits in a hole of A's visible mask, B's box inside A's box
    ms = {0: rect(10, 40, 10, 40, hole=(20, 30, 20, 30)), 1: rect(20, 30, 20, 30)}
    cs = [cand(0, "A", 0.7, ms[0]), cand(1, "B", 0.12, ms[1])]
    r = run(cs, ms, ["A", "B"])
    res["legal_occlusion"] = (r["A"]["status"] == "matched" and r["B"]["status"] == "matched"
                              and r["B"]["cid"] == 1 and not r["B"]["conflicts"])
    # all candidates of B bad
    ms = {0: rect(10, 30, 10, 30), 1: rect(14, 22, 14, 22)}
    cs = [cand(0, "A", 0.7, ms[0]), cand(1, "B", 0.15, ms[1])]
    r = run(cs, ms, ["A", "B"])
    res["all_bad"] = r["A"]["cid"] == 0 and r["A"]["status"] == "matched" and r["B"]["status"] == "missing"
    # ambiguous
    ms = {0: rect(10, 30, 10, 30), 1: rect(40, 48, 2, 10), 2: rect(40, 48, 50, 58)}
    cs = [cand(0, "A", 0.7, ms[0]), cand(1, "B", 0.20, ms[1]), cand(2, "B", 0.20, ms[2])]
    r = run(cs, ms, ["A", "B"])
    res["ambiguous"] = r["B"]["status"] == "ambiguous" and r["A"]["status"] == "matched"
    # near-duplicate candidates of ONE region (other prompts) must not make the object ambiguous
    ms = {0: rect(10, 30, 10, 30), 1: rect(40, 48, 40, 48), 2: rect(40, 48, 40, 49)}
    cs = [cand(0, "A", 0.7, ms[0]), cand(1, "B", 0.20, ms[1]), cand(2, "B", 0.19, ms[2])]
    r = run(cs, ms, ["A", "B"])
    res["duplicate_region_not_ambiguous"] = r["B"]["status"] == "matched" and r["B"]["cid"] == 1
    # part / whole candidates of one object (containment, IoU < 0.7) must not make it ambiguous
    ms = {0: rect(10, 30, 10, 30), 1: rect(40, 50, 40, 50), 2: rect(40, 44, 40, 50)}
    cs = [cand(0, "A", 0.7, ms[0]), cand(1, "B", 0.20, ms[1]), cand(2, "B", 0.19, ms[2])]
    r = run(cs, ms, ["A", "B"])
    res["part_whole_not_ambiguous"] = r["B"]["status"] == "matched" and r["B"]["cid"] == 1
    # order invariance on a 4-object scene with conflicts
    # B's and D's best candidates lie inside A's / C's masks; their alternatives are in free space
    # (the 10-30 version wrongly put B's alternative inside C's mask)
    ms = {0: rect(5, 25, 5, 25), 1: rect(8, 14, 8, 14), 2: rect(40, 46, 2, 8), 3: rect(30, 60, 30, 60),
          4: rect(33, 38, 33, 38), 5: rect(2, 8, 50, 60), 6: rect(52, 62, 2, 12)}
    cs = [cand(0, "A", 0.6, ms[0]), cand(1, "B", 0.30, ms[1]), cand(2, "B", 0.25, ms[2]),
          cand(3, "C", 0.5, ms[3]), cand(4, "D", 0.40, ms[4]), cand(5, "D", 0.20, ms[5]), cand(6, "D", 0.05, ms[6])]
    names = ["A", "B", "C", "D"]
    ref = run(cs, ms, names)
    rng = random.Random(0)
    same = True
    for _ in range(50):
        c2, n2 = cs[:], names[:]
        rng.shuffle(c2)
        rng.shuffle(n2)
        same &= run(c2, ms, n2) == ref
    res["order_invariance"] = (same and ref["B"]["cid"] == 2 and ref["D"]["cid"] == 5
                               and ref["B"]["status"] == ref["D"]["status"] == "matched")
    # object without candidates
    r = run(cs, ms, names + ["E"])
    res["no_candidates"] = r["E"]["status"] == "missing" and all(r[n]["cid"] == ref[n]["cid"] for n in names)
    ok = all(res.values())
    print(" ".join(f"{k}={v}" for k, v in res.items()))
    print("ASSOCIATE_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
