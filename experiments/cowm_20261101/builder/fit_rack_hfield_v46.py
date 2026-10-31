# Copyright 2026 The RLinf Authors.
"""Rack contact-surface fit v46 = v45 (48 public probes, same public observations) with the residual LOCATION fixed.
v45 diagnosis: the model's first-contact point can sit several cm away from the gripper (a finger touches a nearby
part edge first), e.g. acc24_x+060_y+040 and acc24_x+080_y+040 both map to local (.045, -.035) on the -y side, so
residuals were attached to wrong places and neighbouring probes contradicted each other. Here each residual is
attached to the gripper (eef) local xy at the model's first contact, which is set by the action and is the same
point in model and Target. Builder side; public RGB + Builder model only, no GT.
usage: python fit_rack_hfield_v46.py
"""
import json

import numpy as np

import fit_rack_hfield_v42 as H
import fit_rack_hfield_v43 as V

H.SETS["acc45"] = ("rack_accept_dense_plan_v45", "codex_rack_accept_dense_v45")
H.OUT = H.R / "rack_hfield_v45_obs"          # reuse the 48-probe public observations
V.OUT = H.R / "rack_hfield_v46"
V.ITERS = 4


def layer(meas, base_obs):
    lag = float(np.median([r["sim_dec"] - r["sim_first"] for r in meas.values()
                           if r["sim_dec"] is not None and r["sim_first"] is not None]))
    P, Rv = [], []
    for k, r in meas.items():
        o = base_obs[k]["obs_dec"]
        if o is None or o > 240 or r["sim_first"] is None:
            continue
        zf = base_obs[k]["z_free"]
        t = int(np.clip(round(o - lag), 2, H.HI))
        P.append(np.asarray(r["eef_model_local"][r["sim_first"] - 1])[:2])
        Rv.append(zf[t - 1] - zf[r["sim_first"] - 1])
    P, Rv = np.asarray(P), np.asarray(Rv)

    def pred(x, sig, lam, skip=None):
        w = np.exp(-((P - x) ** 2).sum(1) / (2 * sig ** 2))
        if skip is not None:
            w[skip] = 0
        return float((w * Rv).sum() / (w.sum() + lam))
    best = None
    for sig in H.SIGS:
        for lam in H.LAMS:
            e = np.array([abs(pred(P[i], sig, lam, i) - Rv[i]) for i in range(len(Rv))])
            row = (int((e > .02).sum()), float(e.mean()), sig, lam)
            best = row if best is None or row < best else best
    return {"lag": lag, "n": len(Rv), "resid_mean_cm": round(float(np.abs(Rv).mean() * 100), 3),
            "resid_over_2cm": int((np.abs(Rv) > .02).sum()), "loo_over_2cm": best[0],
            "loo_mean_cm": round(best[1] * 100, 3), "sig": best[2], "lam": best[3],
            "location": "eef local xy at model first contact", "P": P, "R": Rv}, pred


V.layer = layer

if __name__ == "__main__":
    V.main()
