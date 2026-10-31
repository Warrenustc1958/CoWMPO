# Copyright 2026 The RLinf Authors.
"""Builder (public RGB + model only): deceleration onset on the v34 far-end probes for given rack candidates.
v33 extent results do not record "states"; they share v31 cand18's initial states (geometry-only change), so an
augmented copy of each result is written under OUT/_inputs (the original result.json is not modified)."""
import json
import sys
from pathlib import Path

import diag_rack_contact_rgb_v31 as D

D.SETS = {"far34": ("rack_far_dense_plan_v34", "codex_rack_far_dense_v34")}
D.OUT = D.R / "diag_rack_far_v34"
STATES = json.loads((D.R / "rack_rib_dev_v31/cand18/result.json").read_text())["states"]


def prepared(c):
    res = json.loads(Path(c).read_text())
    if "states" in res:
        return c
    res["states"] = STATES
    d = D.OUT / "_inputs" / Path(c).parent.parent.name / Path(c).parent.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    return str(d / "result.json")


if __name__ == "__main__":
    for c in sys.argv[1:]:
        D.run(prepared(c))
