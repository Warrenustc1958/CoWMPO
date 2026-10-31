"""S5 debug: per-object pixel counts in the seg render, project/ray_plane round trip, plate mask vs render."""
import os
import sys

import numpy as np

from cowm.init.estimate_state import Scene

world, masks_dir = sys.argv[1], sys.argv[2]
sc = Scene(world)
sc.reset()
seg = sc.seg()
print("objects", sc.names)
print("pixels per obj", {n: int((seg == k).sum()) for k, n in enumerate(sc.names)})
m = sc.env.sim.model
print("ngeom", m.ngeom, "mapped", int((sc.geom2obj >= 0).sum()))
for k, ob in enumerate(sc.env.objects):
    gids = np.nonzero(sc.geom2obj == k)[0]
    print(ob.name, "root", ob.root_body, "geoms", len(gids),
          "groups", sorted(set(int(m.geom_group[g]) for g in gids)))
# round trip
for o in sc.world["objects"]:
    p = np.array(o["pos"][:2] + [sc.world["table"]["height"] + o["top_z"] / 2])
    u, v, _ = sc.env.project(p)
    q = sc.ray_plane(u, v, p[2])
    print(o["name"], "uv", round(u, 1), round(v, 1), "roundtrip err", np.round(q - p, 4))
mk = np.load(os.path.join(masks_dir, "masks.npz"))
for n in mk.files:
    ys, xs = np.nonzero(mk[n])
    print("mask", n, "centroid", round(xs.mean(), 1), round(ys.mean(), 1), "area", len(xs))
