# Copyright 2026 The RLinf Authors.
"""Far-end rack probes v34 (FIT evidence, Builder side, no GT): the S5 static recheck shows a constant x offset that
public frame-0 silhouettes cannot explain (v33: no extension improves public IoU, the far end is not visible). Actively
probe the never-touched far end (x+150 / x+170 / x+190 mm, plus x+170 y+-050) with the same closed-loop vertical
descent as plan_rack_accept_v32.plan(), frames densified like the dense plans (every step 190-260). Target exports
public RGB only; the Builder reads the deceleration onset from public RGB (diag_rack_contact_rgb_v31 rule).
usage: python plan_rack_far_dense_v34.py [--key k]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import plan_rack_accept_v32 as P

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
OUT = R / "rack_far_dense_plan_v34"
SPECS = {f"far34_x{dx:+04d}_y{dy:+04d}": (dx, dy) for dx, dy in
         [(150, 0), (170, 0), (190, 0), (170, -50), (170, 50), (150, -50)]}
WINDOW = (190, 260)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    a = ap.parse_args()
    P.OUT, P.SPECS = OUT, SPECS
    if a.key:
        P.plan(a.key)
        p = json.loads((OUT / a.key / "probe.json").read_text())
        p["frames"] = sorted(set(range(0, len(p["actions"]) + 1, 10)) | set(range(WINDOW[0], WINDOW[1] + 1)))
        p.update(dense_window=list(WINDOW), purpose="fit evidence: far-end active probe (S5 extent), dense public RGB")
        (OUT / a.key / "probe.json").write_text(json.dumps(p, indent=1))
        return
    assert not set(SPECS.values()) & set(P.USED_MM + [(dx, dy) for dx, dy in
                                                      [(40, -60), (60, -60), (110, -60), (50, -40), (90, -40), (70, 40)]])
    OUT.mkdir(exist_ok=False)
    for k in SPECS:
        r = subprocess.run([sys.executable, __file__, "--key", k], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError((k, r.stderr[-800:]))
        print("FAR_PLAN", k, len(json.loads((OUT / k / "probe.json").read_text())["frames"]), flush=True)
    (OUT / "freeze.json").write_text(json.dumps({"keys": sorted(SPECS), "grid_mm": SPECS, "window": WINDOW,
                                                 "GT_inputs": False, "data_roles": "fit evidence (S5 far-end extent)"},
                                                indent=1))
    print("FAR_PLAN_V34_DONE", flush=True)


if __name__ == "__main__":
    main()
