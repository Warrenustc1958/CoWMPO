# Copyright 2026 The RLinf Authors.
"""Rack contact-surface fit v48 = v46 (residuals at eef local xy) with 54 public probes (adds the 6 v47 acceptance
positions, dense public RGB codex_rack_accept_dense_v48, now fit evidence), smaller shrinkage allowed in the LOO grid
(an isolated feature only gets r/(1+lam) per layer) and 5 iterations. Builder side; public RGB + Builder model only.
v47 lesson: one -y feature near local (0, -.066) fails for every model (v44 x+010_y-030, v47 x+000_y-050).
usage: python fit_rack_hfield_v48.py
"""
import json

import fit_rack_hfield_v42 as H
import fit_rack_hfield_v43 as V
import fit_rack_hfield_v46 as F

H.SETS["acc45"] = ("rack_accept_dense_plan_v45", "codex_rack_accept_dense_v45")
H.SETS["acc48"] = ("rack_accept_dense_plan_v48", "codex_rack_accept_dense_v48")
H.OUT = H.R / "rack_hfield_v48_obs"
H.LAMS = (.05, .1, .25, .5, 1., 2.)
V.OUT = H.R / "rack_hfield_v48"
V.ITERS = 5
V.layer = F.layer


def main():
    H.OUT.mkdir(exist_ok=True)
    mp = H.OUT / "meas_base.json"
    if not mp.exists():
        base = json.loads((H.BASE / "result.json").read_text())
        mp.write_text(json.dumps(H.measure(H.BASE / "world", base["states"], True)))
    print("OBS", len(json.loads(mp.read_text())), flush=True)
    V.main()


if __name__ == "__main__":
    main()
