"""S6 debug: G2 replay on a given world/state for a few episodes, per-object step-0 IoU."""
import sys
from cowm.s6.g2_replay import evaluate
world, state = sys.argv[1], sys.argv[2]
for r in evaluate(world, state, "/data_cf/cowmpo/worlds/s6_probe/dev3", "/data_cf/cowmpo/worlds/s6_probe/dev3_masks",
                  ["libero_goal_t0_i0", "libero_goal_t4_i1"]):
    print(r["episode"], "static", round(r["static_iou"], 3), "step0", {k: round(v, 3) for k, v in r["per_frame"]["0"]["iou"].items()})
