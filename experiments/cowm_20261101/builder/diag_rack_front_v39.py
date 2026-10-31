# Copyright 2026 The RLinf Authors.
"""Builder (public RGB + model only): deceleration onset on the v39 dense front-row probes."""
import sys
import diag_rack_far_v34 as F
D = F.D
D.SETS = {"front39": ("rack_front_dense_plan_v39", "codex_rack_front_dense_v39")}
D.OUT = D.R / "diag_rack_front_v39"
if __name__ == "__main__":
    for c in sys.argv[1:]:
        D.run(F.prepared(c))
