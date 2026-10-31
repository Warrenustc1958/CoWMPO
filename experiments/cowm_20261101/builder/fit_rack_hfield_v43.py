# Copyright 2026 The RLinf Authors.
"""Rack contact-surface fit v43 (Builder side; public RGB + Builder model only, no GT, no Target numbers).

v42 lessons: (1) single centre ray per 1 cm cell misses narrow parts (zero-residual world up to 5 steps off the
base model); (2) residuals were measured on the base model but applied to the column representation. v43:
  * prior = max over 5 rays per cell (centre + 4 corners at +-4 mm): closer to the finite gripper footprint;
  * counterexample-guided iterations on the column world itself (WorldCoder / GIF-MCTS style repair loop):
    world_0 = columns at the prior; at iteration k measure first contact on world_k, residual
    r_i = z_free(t_obs) - z_free(first_k), add one smoothed layer (sig, lam by leave-one-probe-out CV), rebuild.
Public observations (obs_dec, z_free) are reused from rack_hfield_v42/meas_base.json. Selection among iterations:
min public transition error with public static IoU unchanged; final judgement only on new unused positions.
usage: python fit_rack_hfield_v43.py
"""
import hashlib
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import fit_rack_hfield_v42 as H

R = H.R
OUT = R / "rack_hfield_v43"
BASE = H.BASE
ITERS = 3
OFFS = [(0., 0.), (.004, .004), (.004, -.004), (-.004, .004), (-.004, -.004)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _prior5(args):
    world, states, cells = args
    import mujoco
    from cowm.init.estimate_state import Scene
    from cowm.s6.plan_probe import apply_state
    sc = Scene(str(world))
    doc = json.loads((Path(states) / H.EP / "state.json").read_text())["objects"]
    robot = json.loads((R / "robot_rgb_all_dev_v2" / H.EP / "robot.json").read_text())["robot"]
    sc.reset()
    apply_state(sc, doc, require_matched=True, robot=robot)
    m, d = sc.env.sim.model, sc.env.sim.data
    mr, dr = getattr(m, "_model", m), getattr(d, "_data", d)
    rack = set(np.flatnonzero(sc.geom2obj == sc.names.index("rack")).tolist())
    bid = int(m.geom_bodyid[min(rack)])
    xpos, xmat = d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()
    gg = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    out = []
    for x, y in cells:
        hs = []
        for dx, dy in OFFS:
            gid = np.array([-1], np.int32)
            dist = mujoco.mj_ray(mr, dr, xpos + xmat @ np.array([x + dx, y + dy, .7]), -xmat[:, 2], gg, 1, -1, gid)
            if dist >= 0 and int(gid[0]) in rack:
                hs.append(.7 - dist)
        out.append(max(hs) if hs else None)
    sc.env.close()
    return out


def grid():
    W, Dd = json.loads((R / "rack_scale_scan_dev_v8/scale00/result.json").read_text())["dims_m"][:2]
    xs = np.arange(-Dd / 2 - .01, Dd / 2 + .08, H.CELL) + H.CELL / 2
    ys = np.arange(-W / 2 - .01, W / 2 + .01, H.CELL) + H.CELL / 2
    return [(float(x), float(y)) for x in xs for y in ys]


def build(name, cells, prior, add):
    dest = OUT / name
    shutil.copytree(BASE / "world", dest / "world")
    xp = dest / "world/assets/rack/rack.xml"
    xml = xp.read_text()
    fr, disabled = None, 0
    for tag in re.findall(r'<geom [^>]*group="0"[^>]*/>', xml):
        top, a = H.geom_top(tag)
        fr = fr or a["friction"]
        if top > H.ZT:
            xml = xml.replace(tag, tag.replace('group="0"', 'group="0" contype="0" conaffinity="0"'), 1)
            disabled += 1
    cols = []
    for (x, y), z, r in zip(cells, prior, add):
        if z is None or z < H.ZT + .003:
            continue
        top = max(z + r, H.ZT + .002)
        cols.append(f'<geom type="box" size="{H.CELL / 2:.5f} {H.CELL / 2:.5f} {(top - H.ZT) / 2:.5f}" '
                    f'pos="{x:.5f} {y:.5f} {(top + H.ZT) / 2:.5f}" group="0" contype="1" conaffinity="1" '
                    f'rgba="0 0 0 0" friction="{fr}" density="0" solref="0.02 1" condim="4"/>')
    assert xml.count("</body><site") == 1
    xp.write_text(xml.replace("</body><site", "".join(cols) + "</body><site", 1))
    meta = json.loads((dest / "world/world.json").read_text())
    meta.pop("world_sha256", None)
    meta["rack_hfield_v43"] = {"prev": str(BASE), "columns": len(cols), "disabled_geoms": disabled,
                               "cell": H.CELL, "zt": H.ZT, "prior": "max of 5 rays"}
    meta["world_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    (dest / "world/world.json").write_text(json.dumps(meta, indent=1))
    return str(dest / "world"), len(cols)


def layer(meas, base_obs):
    """Residuals on the current world and one smoothed layer chosen by leave-one-probe-out CV."""
    lag = float(np.median([r["sim_dec"] - r["sim_first"] for r in meas.values()
                           if r["sim_dec"] is not None and r["sim_first"] is not None]))
    P, Rv, keys = [], [], []
    for k, r in meas.items():
        o = base_obs[k]["obs_dec"]
        if o is None or o > 240 or r["sim_first"] is None:
            continue
        zf = base_obs[k]["z_free"]
        t = int(np.clip(round(o - lag), 2, H.HI))
        P.append(np.asarray(r["contact_local"])[:2]); Rv.append(zf[t - 1] - zf[r["sim_first"] - 1]); keys.append(k)
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
            "loo_mean_cm": round(best[1] * 100, 3), "sig": best[2], "lam": best[3], "P": P, "R": Rv}, pred


def evaluate(world, states, base_obs):
    meas = H.measure(world, states, False)
    err = {k: abs(meas[k]["sim_dec"] - r["obs_dec"]) for k, r in base_obs.items()
           if r["obs_dec"] is not None and r["obs_dec"] <= 240 and meas[k]["sim_dec"] is not None}
    with ProcessPoolExecutor(2, max_tasks_per_child=1) as pool:
        st = sum(pool.map(H._static, [(world, "dev"), (world, "fit")]), [])
    return meas, {"trans_err_mean": round(float(np.mean(list(err.values()))), 3), "trans_err_max": max(err.values()),
                  "n_ge3": int(sum(v >= 3 for v in err.values())), "n": len(err), "per_probe": err,
                  "public_static_iou": round(float(np.mean(st)), 5)}


def main():
    OUT.mkdir(exist_ok=False)
    base = json.loads((BASE / "result.json").read_text())
    obs = json.loads((H.OUT / "meas_base.json").read_text())
    cells = grid()
    prior = _prior5((BASE / "world", base["states"], cells))
    add = np.zeros(len(cells))
    C = np.asarray(cells)
    hist = []
    for it in range(ITERS + 1):
        world, ncol = build(f"iter{it}", cells, prior, add)
        meas, ev = evaluate(world, base["states"], obs)
        rec = {"iter": it, "world": world, "columns": ncol, **ev}
        if it < ITERS:
            L, pred = layer(meas, obs)
            add = add + np.array([pred(c, L["sig"], L["lam"]) for c in C])
            rec["layer"] = {k: v for k, v in L.items() if k not in ("P", "R")}
        hist.append(rec)
        print("ITER", it, json.dumps({k: rec[k] for k in ("trans_err_mean", "trans_err_max", "n_ge3", "n",
                                                          "public_static_iou", "columns")}),
              json.dumps(rec.get("layer", {})), flush=True)
    base_iou = hist[0]["public_static_iou"]
    ok = [h for h in hist if abs(h["public_static_iou"] - base_iou) < 1e-4]
    sel = min(ok, key=lambda h: (h["n_ge3"], h["trans_err_mean"]))
    (OUT / "report.json").write_text(json.dumps({"history": hist, "selected": sel["iter"], "GT_inputs": False,
                                                 "Target_numbers_read": False,
                                                 "rule": "min (#probes >= 3 steps, mean) with static IoU unchanged"},
                                                indent=1))
    print("SELECTED iter", sel["iter"], sel["world"], flush=True)
    print("HFIELD_V43_DONE", flush=True)


if __name__ == "__main__":
    main()
