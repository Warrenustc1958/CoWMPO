"""S5 debug: segmentation-render centroid vs CodeWorldEnv.project() per object."""
import sys
import numpy as np
from cowm.init.estimate_state import Scene
sc = Scene(sys.argv[1]); sc.reset(); seg = sc.seg()
top = sc.world["table"]["height"]
for k, o in enumerate(sc.world["objects"]):
    ys, xs = np.nonzero(seg == k)
    p = sc.env.object_pos(o["name"]) if o["free"] else np.array(o["pos"])
    u, v, _ = sc.env.project([p[0], p[1], top + o["top_z"] / 2])
    print(o["name"], "render", (round(xs.mean(), 1), round(ys.mean(), 1)) if len(xs) else None, "proj", (round(u, 1), round(v, 1)))
