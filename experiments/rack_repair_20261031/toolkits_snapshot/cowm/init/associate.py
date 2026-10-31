"""S5 joint object-candidate association (Builder side; uses no GT and no target internals).

Replaces the greedy rule ("in global score order, each object keeps its first box; a box overlapping an
assigned box is rejected") with one joint assignment over all objects of an episode, with an explicit
"unmatched" option per object. Cost of an assignment (lower is better):
    sum_o unary(o)           unary = -detection score of the chosen candidate, or UNMATCHED_COST
  + sum_{o<p} pair(o, p)     mode "conflict":   LAM * |m_o & m_p| / min(|m_o|, |m_p|)   (pixels explained twice)
                             mode "score_only": +inf if box IoU > MAX_BOX_OVERLAP (the greedy rule), else 0
Exact branch-and-bound over the top-K candidates per object (own prompt, passed the prior filter).
Status per object (abstain=True):
    matched    a candidate is chosen and the best assignment WITHOUT that region (the chosen candidate and all
               own candidates covering the same pixels, containment >= SAME_REGION_FRAC) costs >= MARGIN more
    ambiguous  a candidate is chosen, but a DIFFERENT region is within MARGIN (tentative candidate kept)
    missing    the best assignment leaves the object unmatched
With abstain=False every chosen candidate is "matched" (used for the score-only ablation).
Deterministic and independent of input order: candidates are ranked by (-score, box), objects by name, and
the search only replaces the incumbent on strict improvement.
Known limits: detection scores of different prompts are not calibrated against each other; only the
object's own-prompt candidates are considered (cross-prompt sharing is not implemented); legal overlap is
assumed to show up as disjoint visible masks.
Parameters fixed before evaluation (2026-10-30): TOP_K 12, LAM 1.0, UNMATCHED_COST 0.0, MARGIN 0.05,
SAME_REGION_FRAC 0.7 (containment; changed from IoU after the 10-30 diagnostic).
"""
import numpy as np

TOP_K = 12
LAM = 1.0
UNMATCHED_COST = 0.0
MARGIN = 0.05
MAX_BOX_OVERLAP = 0.6
SAME_REGION_FRAC = 0.7  # candidates of one object are the same region if |a & b| / min(|a|, |b|) >= this
                        # (containment: a part/whole pair is one region; 10-30 v1 used IoU, which split them)
INF = float("inf")


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mask_overlap(a, b):
    """Share of the smaller mask that the other mask also covers (1.0 = one region explained twice)."""
    sa, sb = int(a.sum()), int(b.sum())
    if not sa or not sb:
        return 0.0
    return float(np.logical_and(a, b).sum()) / min(sa, sb)


def mask_iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def options_for(name, candidates, top_k=TOP_K):
    own = [c for c in candidates if c["obj"] == name and c.get("filtered") is None]
    own.sort(key=lambda c: (-float(c["score"]), tuple(round(float(v), 4) for v in c["box"])))
    return own[:top_k]


def _solve(unary, pair, excluded=frozenset()):
    n = len(unary)
    allowed = []
    for i in range(n):
        js = [j for j in range(len(unary[i])) if (i, j) not in excluded]
        js.sort(key=lambda j: (unary[i][j], j))
        allowed.append(js)
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + min(unary[i][j] for j in allowed[i])
    best = [INF, None]
    choice = [0] * n

    def rec(i, cost):
        if i == n:
            if cost < best[0] - 1e-12:
                best[0], best[1] = cost, list(choice)
            return
        for j in allowed[i]:
            c = cost + unary[i][j]
            if c + suffix[i + 1] >= best[0] - 1e-12:
                break                      # options are sorted by unary cost: no later option can do better
            ok = True
            for p in range(i):
                c += pair[p][i][choice[p], j]
                if c + suffix[i + 1] >= best[0] - 1e-12:
                    ok = False
                    break
            if ok:
                choice[i] = j
                rec(i + 1, c)

    rec(0, 0.0)
    return best[0], best[1]


def associate(names, candidates, masks, mode="conflict", abstain=True, top_k=TOP_K, lam=LAM,
              unmatched_cost=UNMATCHED_COST, margin=MARGIN, same_region_frac=SAME_REGION_FRAC):
    """names: world object names; candidates: perceive audit records {cid, obj, score, box, filtered};
    masks: {cid: bool HxW}. Returns {"cost", "mode", "objects": {name: {status, cid, score, margin, alt_cid,
    conflicts}}}."""
    if mode not in ("conflict", "score_only"):
        raise ValueError(f"unknown mode {mode!r}")
    names = sorted(names)
    opts = [[None] + options_for(n, candidates, top_k) for n in names]
    unary = [[unmatched_cost if c is None else -float(c["score"]) for c in o] for o in opts]
    pair = {}
    for p in range(len(names)):
        pair[p] = {}
        for i in range(p + 1, len(names)):
            M = np.zeros((len(opts[p]), len(opts[i])))
            for a, ca in enumerate(opts[p]):
                if ca is None:
                    continue
                for b, cb in enumerate(opts[i]):
                    if cb is None:
                        continue
                    if mode == "conflict":
                        M[a, b] = lam * mask_overlap(masks[ca["cid"]], masks[cb["cid"]])
                    elif box_iou(ca["box"], cb["box"]) > MAX_BOX_OVERLAP:
                        M[a, b] = INF
            pair[p][i] = M
    best, sol = _solve(unary, pair)
    chosen = {names[i]: opts[i][sol[i]] for i in range(len(names))}
    out = {}
    for i, n in enumerate(names):
        c = chosen[n]
        if c is None:
            out[n] = {"status": "missing", "cid": None, "score": None, "margin": None, "alt_cid": None,
                      "conflicts": {}}
            continue
        conf = {}
        for m, cm in chosen.items():
            if m != n and cm is not None:
                v = mask_overlap(masks[c["cid"]], masks[cm["cid"]])
                if v > 0:
                    conf[m] = round(v, 3)
        status, mg, alt_cid = "matched", None, None
        if abstain:
            # margin = how much worse the best assignment gets if this object may NOT use this REGION:
            # exclude the chosen candidate and every own candidate covering the same pixels (near-duplicates
            # from other prompts), so the margin measures whether a DIFFERENT region could explain the object
            mc = masks[c["cid"]]
            excl = frozenset((i, j) for j, cj in enumerate(opts[i])
                             if cj is not None and mask_overlap(masks[cj["cid"]], mc) >= same_region_frac)
            alt_cost, alt_sol = _solve(unary, pair, excl)
            mg = alt_cost - best
            alt = opts[i][alt_sol[i]]
            alt_cid = alt["cid"] if alt is not None else None
            status = "matched" if mg >= margin else "ambiguous"
        out[n] = {"status": status, "cid": c["cid"], "score": round(float(c["score"]), 4),
                  "margin": None if mg is None else round(mg, 4), "alt_cid": alt_cid, "conflicts": conf}
    return {"cost": best, "mode": mode, "objects": out}
