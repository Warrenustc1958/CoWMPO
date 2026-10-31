# Copyright 2026 The RLinf Authors.
"""Rack contact-surface fit v42 (Builder side; public RGB + Builder model only, no GT, no Target numbers).

Hand-designed part families (v17-v41) fixed one probe set and failed on the next. Here, as in transition-level
code-world-model repair (WorldCoder / PatchWorld style), the judged transition itself is fitted:
  * visual geometry of v36 cand03 is unchanged (static silhouette cannot change);
  * upper-tier collision geoms (top > ZT, local) are disabled; invisible collision columns (group 0, rgba alpha 0; group 3 is dropped by the loader, 1 cm cells,
    ZT .. top) carry the top surface. top = prior (ray-cast on v36c03) + residual field r(x, y);
  * per public probe: observed first contact t_obs = public-RGB deceleration onset - lag (lag = median
    sim_dec - sim_first on the model); residual r_i = z_free(t_obs) - z_free(sim_first), z_free = Builder eef height
    with the rack moved away (same actions); location = model contact point (rack local);
  * r(x) = sum w r_i / (sum w + lam), w = exp(-|x - x_i|^2 / 2 sig^2); (sig, lam) by leave-one-probe-out CV.
Probes excluded by the fixed rule: obs_dec None or > 240 (no blocked moment, slides).
usage: python fit_rack_hfield_v42.py            (measure -> fit -> build -> evaluate)
"""
import hashlib
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import diag_rack_contact_rgb_v31 as D

R = D.R
OUT = R / "rack_hfield_v42"
BASE = R / "rack_far_dev_v36/cand03"
SETS = dict(D.SETS)
SETS.update(far34=("rack_far_dense_plan_v34", "codex_rack_far_dense_v34"),
            front39=("rack_front_dense_plan_v39", "codex_rack_front_dense_v39"),
            acc42=("rack_accept_dense_plan_v42", "codex_rack_accept_dense_v42"))
EP, LO, HI = D.EP, D.LO, D.HI
CELL, ZT = .01, .22
SIGS, LAMS = (.015, .02, .03, .04, .06), (.25, .5, 1., 2.)
DIRS = {"dev": ("natural_states_all_dev_v1", "robot_rgb_all_dev_v2", "stove_parts_states_dev_v1"),
        "fit": ("natural_bowl_fit_v2", "robot_rgb_fit_v2", "stove_parts_states_fit_v1")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def probes():
    for s, (plan, _) in SETS.items():
        for d in sorted((R / plan).iterdir()):
            if (d / "probe.json").exists():
                yield s, d.name


def _measure(args):
    world, states, s, key, free = args
    from cowm.init.estimate_state import Scene
    from cowm.s6.plan_probe import apply_state
    plan, ev = SETS[s]
    probe = json.loads((R / plan / key / "probe.json").read_text())
    sc = Scene(str(world))
    m, d = sc.env.sim.model, sc.env.sim.data
    k = sc.names.index("rack")
    grip = np.array([(m.body_id2name(int(m.geom_bodyid[g])) or "").startswith("gripper0") for g in range(m.ngeom)])
    doc = json.loads((Path(states) / EP / "state.json").read_text())["objects"]
    robot = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    bid = int(m.geom_bodyid[np.flatnonzero(sc.geom2obj == k)[0]])
    steps = list(range(LO, HI + 1))

    def run(away):
        sc.reset()
        apply_state(sc, doc, require_matched=True, robot=robot)
        if away:
            o = doc["rack"]
            sc.set_pose(k, [o["pos"][0] + 3, o["pos"][1] + 3], o["yaw"], o["pos"][2])
        xpos, xmat = d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()
        eef, first, cpos, prev, ch = [], None, None, None, []
        for step, act in enumerate(probe["actions"], 1):
            sc.env.step(np.asarray(act))
            eef.append([float(v) for v in sc.env.eef_pos()])
            if not away and first is None:
                for c in d.contact[:d.ncon]:
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if (grip[g1] and sc.geom2obj[g2] == k) or (grip[g2] and sc.geom2obj[g1] == k):
                        first, cpos = step, np.array(c.pos, float)
                        break
            if not away and step >= LO - 1:
                mk = sc.robot_mask()
                if prev is not None:
                    ch.append(int((mk ^ prev).sum()))
                prev = mk
            if step >= HI:
                break
        return eef, first, cpos, ch, xpos, xmat

    eef, first, cpos, ch, xpos, xmat = run(False)
    rec = {"set": s, "key": key, "sim_first": first, "sim_dec": D.decel(ch, steps, .5),
           "contact_local": None if cpos is None else (xmat.T @ (cpos - xpos)).tolist(),
           "eef_model_local": [(xmat.T @ (np.asarray(e) - xpos)).tolist() for e in eef]}
    if free:
        fe = run(True)[0]
        rec["z_free"] = [e[2] for e in fe]
        rec["obs_dec"] = D.decel(D.obs_change(D.EV / ev / key, steps), steps, .5)
    sc.env.close()
    return rec


def measure(world, states, free):
    jobs = [(world, states, s, k, free) for s, k in probes()]
    with ProcessPoolExecutor(len(jobs), max_tasks_per_child=1) as pool:
        return {r["key"]: r for r in pool.map(_measure, jobs)}


def fit(meas):
    lag = float(np.median([r["sim_dec"] - r["sim_first"] for r in meas.values()
                           if r["sim_dec"] is not None and r["sim_first"] is not None]))
    offs = [np.asarray(r["contact_local"])[:2] - np.asarray(r["eef_model_local"][r["sim_first"] - 1])[:2]
            for r in meas.values() if r["sim_first"] is not None]
    off = np.median(offs, axis=0)
    P, Rv, V, keys, excluded = [], [], [], [], {}
    for key, r in meas.items():
        o = r["obs_dec"]
        if o is None or o > 240:
            excluded[key] = "obs_dec None or > 240"
            continue
        if r["sim_first"] is None:
            excluded[key] = "model has no contact"
            continue
        t = int(np.clip(round(o - lag), 2, HI))
        zf = r["z_free"]
        P.append(np.asarray(r["contact_local"])[:2])
        Rv.append(zf[t - 1] - zf[r["sim_first"] - 1])
        V.append(max(abs(zf[t - 1] - zf[t - 2]), 1e-4))
        keys.append(key)
    P, Rv, V = np.asarray(P), np.asarray(Rv), np.asarray(V)

    def pred(x, sig, lam, skip=None):
        w = np.exp(-((P - x) ** 2).sum(1) / (2 * sig ** 2))
        if skip is not None:
            w[skip] = 0
        return float((w * Rv).sum() / (w.sum() + lam))
    table = []
    base_err = np.abs(Rv)
    for sig in SIGS:
        for lam in LAMS:
            e = np.array([abs(pred(P[i], sig, lam, i) - Rv[i]) for i in range(len(Rv))])
            table.append({"sig": sig, "lam": lam, "loo_mean_cm": round(float(e.mean() * 100), 3),
                          "loo_max_cm": round(float(e.max() * 100), 3),
                          "loo_over_2cm": int((e > .02).sum()),
                          "loo_mean_steps": round(float((e / V).mean()), 3),
                          "loo_max_steps": round(float((e / V).max()), 3)})
    best = min(table, key=lambda t: (t["loo_over_2cm"], t["loo_mean_cm"]))
    return {"lag": lag, "xy_offset": off.tolist(), "n_used": len(keys), "excluded": excluded,
            "residual_cm": {k: round(float(v * 100), 3) for k, v in zip(keys, Rv)},
            "baseline_mean_cm": round(float(base_err.mean() * 100), 3),
            "baseline_max_cm": round(float(base_err.max() * 100), 3),
            "baseline_over_2cm": int((base_err > .02).sum()),
            "loo_table": table, "selected": best, "P": P.tolist(), "R": Rv.tolist()}


def _prior(args):
    world, states, cells = args
    import mujoco
    from cowm.init.estimate_state import Scene
    from cowm.s6.plan_probe import apply_state
    sc = Scene(str(world))
    doc = json.loads((Path(states) / EP / "state.json").read_text())["objects"]
    robot = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    sc.reset()
    apply_state(sc, doc, require_matched=True, robot=robot)
    m, d = sc.env.sim.model, sc.env.sim.data
    mr, dr = getattr(m, "_model", m), getattr(d, "_data", d)
    k = sc.names.index("rack")
    rack = set(np.flatnonzero(sc.geom2obj == k).tolist())
    bid = int(m.geom_bodyid[min(rack)])
    xpos, xmat = d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()
    gg = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    out = []
    for x, y in cells:
        p = xpos + xmat @ np.array([x, y, .7])
        gid = np.array([-1], np.int32)
        dist = mujoco.mj_ray(mr, dr, p, -xmat[:, 2], gg, 1, -1, gid)
        out.append(float(.7 - dist) if dist >= 0 and int(gid[0]) in rack else None)
    sc.env.close()
    return out


def geom_top(tag):
    a = dict(re.findall(r'(\w+)="([^"]*)"', tag))
    assert a["type"] == "box", tag
    s, p = [float(v) for v in a["size"].split()], [float(v) for v in a["pos"].split()]
    e = [float(v) for v in a.get("euler", "0 0 0").split()]
    assert e[0] == 0 and e[2] == 0, tag
    return p[2] + s[2] * abs(np.cos(e[1])) + s[0] * abs(np.sin(e[1])), a


def build(name, fitres, sig, lam, zero):
    base = json.loads((BASE / "result.json").read_text())
    W, Dd = json.loads((R / "rack_scale_scan_dev_v8/scale00/result.json").read_text())["dims_m"][:2]
    xs = np.arange(-Dd / 2 - .01, Dd / 2 + .08, CELL) + CELL / 2
    ys = np.arange(-W / 2 - .01, W / 2 + .01, CELL) + CELL / 2
    cells = [(float(x), float(y)) for x in xs for y in ys]
    prior = _prior((BASE / "world", base["states"], cells))
    P, Rv = np.asarray(fitres["P"]), np.asarray(fitres["R"])
    dest = OUT / name
    shutil.copytree(BASE / "world", dest / "world")
    xp = dest / "world/assets/rack/rack.xml"
    xml = xp.read_text()
    fr, disabled = None, 0
    for tag in re.findall(r'<geom [^>]*group="0"[^>]*/>', xml):
        top, a = geom_top(tag)
        fr = fr or a["friction"]
        if top > ZT:
            xml = xml.replace(tag, tag.replace('group="0"', 'group="0" contype="0" conaffinity="0"'), 1)
            disabled += 1
    cols, n = [], 0
    for (x, y), z in zip(cells, prior):
        if z is None or z < ZT + .003:
            continue
        r = 0. if zero else float((np.exp(-((P - [x, y]) ** 2).sum(1) / (2 * sig ** 2)) * Rv).sum() /
                                  (np.exp(-((P - [x, y]) ** 2).sum(1) / (2 * sig ** 2)).sum() + lam))
        top = z + r
        cols.append(f'<geom type="box" size="{CELL / 2:.5f} {CELL / 2:.5f} {(top - ZT) / 2:.5f}" '
                    f'pos="{x:.5f} {y:.5f} {(top + ZT) / 2:.5f}" group="0" contype="1" conaffinity="1" rgba="0 0 0 0" friction="{fr}" '
                    f'density="0" solref="0.02 1" condim="4"/>')
        n += 1
    assert xml.count("</body><site") == 1
    xml = xml.replace("</body><site", "".join(cols) + "</body><site", 1)
    xp.write_text(xml)
    meta = json.loads((dest / "world/world.json").read_text())
    next(o for o in meta["objects"] if o["name"] == "rack")["attrs"].update(
        hfield_cell=CELL, hfield_zt=ZT, hfield_sig=sig, hfield_lam=lam, hfield_zero=zero)
    meta.pop("world_sha256", None)
    meta["rack_hfield_v42"] = {"prev": str(BASE), "sig": sig, "lam": lam, "zero_residual": zero, "columns": n,
                               "disabled_geoms": disabled}
    meta["world_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    (dest / "world/world.json").write_text(json.dumps(meta, indent=1))
    return {"world": str(dest / "world"), "columns": n, "disabled_geoms": disabled}


def _static(args):
    world, role = args
    from cowm.init.estimate_state import Scene, iou
    from cowm.s6.plan_probe import apply_state
    sd, rd, md = DIRS[role]
    sc = Scene(str(world))
    k = sc.names.index("rack")
    v = []
    for ep in sorted(p.parent.name for p in (R / sd).glob("*/state.json")):
        sc.reset()
        apply_state(sc, json.loads((R / sd / ep / "state.json").read_text())["objects"], require_matched=True,
                    robot=json.loads((R / rd / ep / "robot.json").read_text())["robot"])
        with np.load(R / md / ep / "masks.npz") as a:
            v.append(float(iou(sc.seg() == k, a["rack"])))
    sc.env.close()
    return v


def evaluate(world, states, meas_base):
    meas = measure(world, states, False)
    err = {k: abs(meas[k]["sim_dec"] - r["obs_dec"]) for k, r in meas_base.items()
           if r["obs_dec"] is not None and r["obs_dec"] <= 240 and meas[k]["sim_dec"] is not None}
    with ProcessPoolExecutor(2, max_tasks_per_child=1) as pool:
        st = sum(pool.map(_static, [(world, "dev"), (world, "fit")]), [])
    return {"trans_err_mean": round(float(np.mean(list(err.values()))), 3), "trans_err_max": max(err.values()),
            "n": len(err), "per_probe": err, "sim_dec": {k: v["sim_dec"] for k, v in meas.items()},
            "public_static_iou": round(float(np.mean(st)), 5)}


def main():
    OUT.mkdir(exist_ok=True)
    base = json.loads((BASE / "result.json").read_text())
    mp = OUT / "meas_base.json"
    if not mp.exists():
        mp.write_text(json.dumps(measure(BASE / "world", base["states"], True)))
    meas = json.loads(mp.read_text())
    f = fit(meas)
    (OUT / "fit.json").write_text(json.dumps({k: v for k, v in f.items() if k not in ("P", "R")}, indent=1))
    print("FIT", json.dumps({k: f[k] for k in ("lag", "n_used", "excluded", "baseline_mean_cm", "baseline_max_cm",
                                               "baseline_over_2cm", "selected")}), flush=True)
    sel = f["selected"]
    worlds = {"zero": build("zero_g0", f, sel["sig"], sel["lam"], True),
              "fit": build("fit_g0", f, sel["sig"], sel["lam"], False)}
    rep = {"base": evaluate(BASE / "world", base["states"], meas)}
    for name, w in worlds.items():
        rep[name] = {**w, **evaluate(w["world"], base["states"], meas)}
    rep["zero_vs_base_max_dec_diff"] = max(abs((rep["zero"]["sim_dec"][k] or 0) - (rep["base"]["sim_dec"][k] or 0))
                                          for k in rep["base"]["sim_dec"])
    (OUT / "report_g0.json").write_text(json.dumps({"fit": {k: v for k, v in f.items() if k not in ("P", "R")},
                                                 **rep, "GT_inputs": False, "Target_numbers_read": False}, indent=1))
    for name in ("base", "zero", "fit"):
        r = rep[name]
        print("EVAL", name, json.dumps({k: r[k] for k in ("trans_err_mean", "trans_err_max", "n", "public_static_iou")
                                        if k in r}), r.get("columns"), r.get("disabled_geoms"), flush=True)
    print("ZERO_VS_BASE_MAX_DEC_DIFF", rep["zero_vs_base_max_dec_diff"], flush=True)
    print("HFIELD_V42_DONE", flush=True)


if __name__ == "__main__":
    main()
