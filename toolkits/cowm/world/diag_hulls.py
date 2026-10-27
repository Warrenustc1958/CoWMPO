"""S4 diagnostic: CoACD collision hulls of revolved templates (z range, thickness, footprint)."""
import glob
import os

import numpy as np
import trimesh

OUT = os.environ.get("COWM_WORLDS", "/data_cf/cowmpo/worlds")


def main():
    for obj in ("plate", "bowl"):
        d = os.path.join(OUT, "s4_ref_goal_v0", "assets", obj)
        vis = trimesh.load(os.path.join(d, f"{obj}_vis.obj"), force="mesh")
        print(f"== {obj}: visual z [{vis.bounds[0][2]*1000:.1f}, {vis.bounds[1][2]*1000:.1f}] mm, "
              f"watertight={vis.is_watertight}, volume={vis.volume*1e6:.1f} cm3")
        for p in sorted(glob.glob(os.path.join(d, f"{obj}_c*.obj"))):
            h = trimesh.load(p, force="mesh")
            lo, hi = h.bounds
            r = np.linalg.norm(h.vertices[:, :2], axis=1)
            print(f"   {os.path.basename(p):12s} z [{lo[2]*1000:5.1f}, {hi[2]*1000:5.1f}] mm  "
                  f"r [{r.min()*1000:5.1f}, {r.max()*1000:5.1f}] mm  vol={h.volume*1e6:6.2f} cm3")


if __name__ == "__main__":
    main()
