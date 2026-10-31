# Copyright 2026 The RLinf Authors.
"""Rack contact-surface fit v45 = v43 procedure with 48 public probes (adds the 6 v44 acceptance positions, dense
public RGB codex_rack_accept_dense_v45, now fit evidence). Builder side; public RGB + Builder model only, no GT.
v44 lesson: the LOO-selected kernel is narrow (1.5 cm), so the correction only acts near probes; more probes, same
method. Public observations are re-measured for all 48 probes into rack_hfield_v45_obs (v42 cache untouched).
usage: python fit_rack_hfield_v45.py
"""
import json

import fit_rack_hfield_v42 as H
import fit_rack_hfield_v43 as V

H.SETS["acc45"] = ("rack_accept_dense_plan_v45", "codex_rack_accept_dense_v45")
H.OUT = H.R / "rack_hfield_v45_obs"
V.OUT = H.R / "rack_hfield_v45"
V.ITERS = 4


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
