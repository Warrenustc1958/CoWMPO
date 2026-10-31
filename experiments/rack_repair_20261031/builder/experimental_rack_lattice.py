# Copyright 2026 The RLinf Authors.
"""Open-lattice rack hypothesis (same 4 parameters as the slatted two-shelf model: width, depth, height, incline).

Thin rods instead of solid slats/plates: base = 4 perimeter rails, each shelf = 4 perimeter rails + 3 cross
rods, 4 corner posts. Built only from these parameters; no target geometry. parts() returns the same box list
(half, pos, pitch) for the silhouette fitter as build() writes to MuJoCo.
"""
import math

from cowm.world.templates import _part, _save, _wrap

ROD = 0.006          # rod half thickness (1.2 cm square rods)
CROSS = 3            # cross rods per shelf


def parts(width, depth, height, incline):
    slope, co = math.sin(incline), math.cos(incline)
    upper = height - .45 * depth * abs(slope) - ROD
    lower = max(.05, .45 * upper)
    rows = []
    for x in (-depth / 2 + ROD, depth / 2 - ROD):                      # base rails along y
        rows.append(([ROD, width / 2, ROD], [x, 0, ROD], 0.))
    for y in (-width / 2 + ROD, width / 2 - ROD):                      # base rails along x
        rows.append(([depth / 2, ROD, ROD], [0, y, ROD], 0.))
    for level in (lower, upper):
        for x in (-.45 * depth, .45 * depth):                          # shelf front/back rails (along y)
            rows.append(([ROD, width / 2, ROD], [co * x, 0, level - slope * x], incline))
        for y in (-width / 2 + ROD, width / 2 - ROD):                  # shelf side rails (along x, inclined)
            rows.append(([.45 * depth, ROD, ROD], [0, y, level], incline))
        for j in range(CROSS):                                         # cross rods (along y)
            x = (-.45 + .9 * (j + 1) / (CROSS + 1)) * depth
            rows.append(([ROD, width / 2, ROD], [co * x, 0, level - slope * x], incline))
    for x in (-depth / 2 + ROD, depth / 2 - ROD):                      # corner posts
        for y in (-width / 2 + ROD, width / 2 - ROD):
            rows.append(([ROD, ROD, height / 2], [x, y, height / 2], 0.))
    return [(__import__("numpy").array(h), __import__("numpy").array(p), a) for h, p, a in rows]


def build(name, out_dir, width, depth, height, incline, density=600., friction=.7):
    rgba = '0.45 0.3 0.2 1'
    body = ''
    for half, pos, pitch in parts(width, depth, height, incline):
        body += _part('box', [float(v) for v in half], [float(v) for v in pos], rgba,
                      euler=f'0 {pitch} 0' if pitch else None, density=density, friction=friction)
    return _save(out_dir, name, _wrap(name, '', body, height, max(width, depth) / 2))
