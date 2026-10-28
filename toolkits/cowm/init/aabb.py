"""World-frame axis-aligned bounding box of an object's geoms (shared by Builder and Target-side evaluation).

Uses MuJoCo's per-geom local AABB (model.geom_aabb: center xyz + half sizes xyz in the geom frame) and the
geom pose, so the result does not depend on where an asset puts its body origin.
"""
import numpy as np


def object_aabb(model, data, geom_ids):
    """(center[3], half_extent[3]) of the union of the given geoms' boxes in the world frame."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    for g in geom_ids:
        a = np.asarray(model.geom_aabb[g], float)
        c, h = a[:3], a[3:]
        if not np.all(np.isfinite(h)) or np.all(h == 0):
            continue
        R = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        pts = data.geom_xpos[g] + (c + signs * h) @ R.T
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    if not np.all(np.isfinite(lo)):
        return None, None
    return (lo + hi) / 2, (hi - lo) / 2
