"""CoWM-PO verifier for ANY world program (run inside the Builder sandbox).

lint -> compile -> footprint/table -> settle -> zero-action stillness -> determinism -> reach -> articulation
-> camera visibility (+ agentview render). Writes a JSON report the Agent reads on failure.
Physical FIDELITY to the target is NOT checked here (that is S6/S8); this only checks the world is
self-consistent and usable for the tasks.
"""
import argparse
import inspect
import json
import math
import os
import traceback

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np  # noqa: E402

from cowm.world.compile_world import compile_world  # noqa: E402
from cowm.world.lint import LintError, lint  # noqa: E402
from cowm.world.templates import TEMPLATES  # noqa: E402

GATES = {"settle_mm": 10.0, "drift_mm": 1.0, "min_contact_mm": -3.0, "reach_m": 0.05,
         "artic_frac": 0.3, "img_margin_px": 4, "overlap_m": 0.002}
HOLD = np.array([0, 0, 0, 0, 0, 0, -1.0])


def _attrs(o):
    d = {k: v.default for k, v in inspect.signature(TEMPLATES[o["category"]]).parameters.items()
         if v.default is not inspect.Parameter.empty}
    d.update(o["attrs"])
    return d


def _half_extents(o):
    a, c = _attrs(o), o["category"]
    if c in ("bowl", "plate", "bottle"):
        return a["radius"], a["radius"]
    if c == "box":
        return a["size_x"] / 2, a["size_y"] / 2
    if c == "drawer_cabinet":
        return a["depth"] / 2 + 0.02, a["width"] / 2           # + handles at the front
    if c == "stove":
        return a["depth"] / 2 + 0.025, a["width"] / 2          # + knob at the front
    return a["depth"] / 2, a["width"] / 2


def check_footprints(world):
    errs, boxes = [], []
    tx, ty = world["table"]["size"][0] / 2, world["table"]["size"][1] / 2
    for o in world["objects"]:
        hx, hy = _half_extents(o)
        cs, sn = abs(math.cos(o["yaw"])), abs(math.sin(o["yaw"]))
        ex, ey = cs * hx + sn * hy, sn * hx + cs * hy
        x, y = o["pos"][:2]
        boxes.append((o["name"], x - ex, x + ex, y - ey, y + ey))
        if abs(x) + ex > tx + 1e-6 or abs(y) + ey > ty + 1e-6:
            errs.append(f"object '{o['name']}' footprint [{x - ex:.3f},{x + ex:.3f}]x[{y - ey:.3f},{y + ey:.3f}] "
                        f"leaves the table [-{tx:.2f},{tx:.2f}]x[-{ty:.2f},{ty:.2f}]")
    g = GATES["overlap_m"]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ox = min(a[2], b[2]) - max(a[1], b[1])
            oy = min(a[4], b[4]) - max(a[3], b[3])
            if ox > g and oy > g:
                both_fixed = not world["objects"][i]["free"] and not world["objects"][j]["free"]
                note = (" (fixed fixtures do not collide in simulation, so this is checked geometrically)"
                        if both_fixed else "")
                errs.append(f"footprints of '{a[0]}' and '{b[0]}' overlap by {ox * 1000:.0f} x {oy * 1000:.0f} mm"
                            + note)
    return errs


def _drive(env, target, steps):
    for _ in range(steps):
        e = env.eef_pos()
        a = HOLD.copy()
        a[:3] = np.clip((np.asarray(target) - e) / 0.05, -1, 1)
        env.step(a)
    return float(np.linalg.norm(env.eef_pos() - np.asarray(target)))


def reach_targets(world, top):
    tg = []
    for o in world["objects"]:
        x, y = o["pos"][:2]
        tg.append((f"{o['name']} (top)", [x, y, top + o["top_z"] + 0.04], None))
        if o["category"] == "drawer_cabinet":
            a = _attrs(o)
            n, H = int(a["n_drawers"]), a["height"]
            hd = (H - 0.02) / n
            z = top + 0.01 + (n // 2) * hd + 0.003 + (hd - 0.012) / 2
            fx, fy = -math.cos(o["yaw"]), -math.sin(o["yaw"])            # front direction (object -x)
            d = a["depth"] / 2 + 0.03
            tg.append((f"{o['name']} (drawer handle)", [x + fx * d, y + fy * d, z], [x + fx * (d + 0.12),
                                                                                       y + fy * (d + 0.12), z]))
    return tg


def check_reach(env, world):
    top, res, errs = world["table"]["height"], {}, []
    for name, target, approach in reach_targets(world, top):
        env.reset()
        for _ in range(5):
            env.step(HOLD)
        safe_z = max(target[2], top + 0.45)
        _drive(env, [env.eef_pos()[0], env.eef_pos()[1], safe_z], 30)
        if approach is not None:
            _drive(env, approach, 50)
        else:
            _drive(env, [target[0], target[1], safe_z], 50)
        dist = _drive(env, target, 50)
        res[name] = round(dist, 4)
        if dist > GATES["reach_m"]:
            errs.append(f"robot cannot reach {name}: end-effector stays {dist * 100:.0f} cm from target "
                        f"{np.round(target, 3).tolist()} (move it closer to the robot base or lower it)")
    return res, errs


def check_articulation(env, world):
    m, d = env.sim.model, env.sim.data
    res, errs = {}, []
    for o in world["objects"]:
        for j in o["joints"]:
            jn = f"{o['name']}_{j}"
            jid = m.joint_name2id(jn)
            dof, qa = m.jnt_dofadr[jid], m.jnt_qposadr[jid]
            lo, hi = m.jnt_range[jid]
            env.reset()
            q0 = float(d.qpos[qa])
            sign = 1.0 if (hi - q0) > (q0 - lo) else -1.0
            mag = 20.0 if m.jnt_type[jid] == 2 else 0.02           # N for slide, N*m for hinge (light knob)
            d.qfrc_applied[dof] = sign * mag
            for _ in range(40):
                env.step(HOLD)
            d.qfrc_applied[dof] = 0.0
            q1 = float(d.qpos[qa])
            frac = abs(q1 - q0) / max(hi - lo, 1e-9)
            res[jn] = round(frac, 3)
            if q1 < lo - 0.1 * (hi - lo) or q1 > hi + 0.1 * (hi - lo):
                errs.append(f"joint '{jn}' was pushed {frac:.0%} of its range, beyond its limits "
                            f"[{lo:.3f}, {hi:.3f}] (joint limit not holding)")
            elif frac < GATES["artic_frac"]:
                errs.append(f"joint '{jn}' moved only {frac:.0%} of its range under a {mag} push "
                            f"(blocked by geometry or damping/frictionloss too high)")
    env.reset()
    return res, errs


def check_camera(env, world):
    for _ in range(15):
        env.step(HOLD)
    H, W = world["camera"]["image_size"]
    mg, res, errs = GATES["img_margin_px"], {}, []
    top = world["table"]["height"]
    for o in world["objects"]:
        p = env.object_pos(o["name"]) if o["free"] else np.array(o["pos"])
        p = np.array([p[0], p[1], top + o["top_z"] / 2])
        u, v, depth = env.project(p)
        res[o["name"]] = [round(float(u), 1), round(float(v), 1)]
        if depth <= 0 or not (mg <= u <= W - 1 - mg and mg <= v <= H - 1 - mg):
            errs.append(f"object '{o['name']}' is outside the agentview image (pixel {u:.0f},{v:.0f} in {W}x{H})")
    return res, errs


def render_agentview(wdir, out_png):
    from PIL import Image

    from cowm.world.env import CodeWorldEnv

    env = CodeWorldEnv(wdir, use_camera_obs=True, camera_names="agentview", camera_heights=256,
                       camera_widths=256, has_renderer=False, has_offscreen_renderer=True)
    obs = env.reset()
    for _ in range(15):
        obs, _, _, _ = env.step(HOLD)
    Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).save(out_png)


def check(program, out_dir):
    rep = {"program": program, "stage": None, "ok": False, "errors": []}
    src = open(program).read()
    try:
        lint(src)
    except (LintError, SyntaxError) as e:
        rep.update(stage="lint", errors=[str(e)])
        return rep
    try:
        wdir = compile_world(program, out_dir)
    except Exception as e:  # noqa: BLE001
        rep.update(stage="compile", errors=[f"{type(e).__name__}: {e}", traceback.format_exc(limit=3)])
        return rep
    world = json.load(open(os.path.join(wdir, "world.json")))
    rep["world"] = {"objects": [(o["name"], o["category"], o["free"], o["material"]) for o in world["objects"]],
                    "params": list(world["params"]), "world_sha256": world["world_sha256"]}
    errs = check_footprints(world)
    try:
        from cowm.world.env import CodeWorldEnv

        env = CodeWorldEnv(wdir, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    except Exception as e:  # noqa: BLE001
        rep.update(stage="load", errors=errs + [f"{type(e).__name__}: {e}"])
        return rep
    free = [o["name"] for o in world["objects"] if o["free"]]
    env.reset()
    p0 = {n: env.object_pos(n) for n in free}
    for _ in range(80):
        env.step(HOLD)
    p1 = {n: env.object_pos(n) for n in free}
    for _ in range(20):
        env.step(HOLD)
    p2 = {n: env.object_pos(n) for n in free}
    m, d = env.sim.model, env.sim.data
    worst = (0.0, "", "")
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist * 1000 < worst[0]:
            worst = (c.dist * 1000, m.body_id2name(m.geom_bodyid[c.geom1]), m.body_id2name(m.geom_bodyid[c.geom2]))
    settle = {n: float(np.linalg.norm(p1[n] - p0[n]) * 1000) for n in free}
    drift = {n: float(np.linalg.norm(p2[n] - p1[n]) * 1000) for n in free}
    for n in free:
        if settle[n] > GATES["settle_mm"]:
            errs.append(f"object '{n}' moved {settle[n]:.1f} mm while settling (gate {GATES['settle_mm']}): "
                        f"likely placed overlapping another object or off the table")
        if drift[n] > GATES["drift_mm"]:
            errs.append(f"object '{n}' still moving {drift[n]:.2f} mm/s after settling: unstable placement")
    if worst[0] < GATES["min_contact_mm"]:
        errs.append(f"penetration {worst[0]:.1f} mm between '{worst[1]}' and '{worst[2]}': objects overlap")
    acts = np.clip(np.random.default_rng(0).normal(0, 0.3, size=(30, 7)), -1, 1)
    traj = []
    for _ in range(2):
        env.reset()
        traj.append(np.stack([(env.step(a), env.sim.data.qpos.copy())[1] for a in acts]))
    if float(np.abs(traj[0] - traj[1]).max()) != 0.0:
        errs.append("world is not deterministic under identical actions")
    env.reset()
    cam_res, cam_err = check_camera(env, world)
    reach_res, reach_err = check_reach(env, world)
    art_res, art_err = check_articulation(env, world)
    errs += cam_err + reach_err + art_err
    rep["physics"] = {"settle_mm": settle, "drift_mm": drift, "worst_contact": worst}
    rep["usability"] = {"reach_dist_m": reach_res, "articulation_range_frac": art_res, "image_px": cam_res}
    try:
        render_agentview(wdir, os.path.join(out_dir, "agentview.png"))
        rep["render"] = os.path.join(out_dir, "agentview.png")
    except Exception as e:  # noqa: BLE001
        rep["render_error"] = f"{type(e).__name__}: {e}"
    rep.update(stage="checks", errors=errs, ok=not errs)
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--program", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rep = check(a.program, a.out)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "check_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print("CHECK", "PASS" if rep["ok"] else f"FAIL[{rep['stage']}]", json.dumps(rep["errors"])[:600])


if __name__ == "__main__":
    main()
