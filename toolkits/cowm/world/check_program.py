"""CoWM-PO verifier for ANY world program (run inside the Builder sandbox).

lint -> compile -> settle -> zero-action stillness -> determinism. Writes a JSON report that the Agent
reads on failure. Physical fidelity to the target is NOT checked here (that is S6/S8).
"""
import argparse
import json
import os
import traceback

import numpy as np

from cowm.world.compile_world import compile_world
from cowm.world.lint import LintError, lint

GATES = {"settle_mm": 10.0, "drift_mm": 1.0, "min_contact_mm": -3.0}


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
    rep["world"] = {"objects": [(o["name"], o["category"], o["free"]) for o in world["objects"]],
                    "params": list(world["params"]), "world_sha256": world["world_sha256"]}
    try:
        from cowm.world.env import CodeWorldEnv

        env = CodeWorldEnv(wdir, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    except Exception as e:  # noqa: BLE001
        rep.update(stage="load", errors=[f"{type(e).__name__}: {e}"])
        return rep
    free = [o["name"] for o in world["objects"] if o["free"]]
    hold = np.zeros(7)
    hold[6] = -1.0
    env.reset()
    p0 = {n: env.object_pos(n) for n in free}
    for _ in range(80):
        env.step(hold)
    p1 = {n: env.object_pos(n) for n in free}
    for _ in range(20):
        env.step(hold)
    p2 = {n: env.object_pos(n) for n in free}
    m, d = env.sim.model, env.sim.data
    worst = (0.0, "", "")
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist * 1000 < worst[0]:
            worst = (c.dist * 1000, m.body_id2name(m.geom_bodyid[c.geom1]), m.body_id2name(m.geom_bodyid[c.geom2]))
    settle = {n: float(np.linalg.norm(p1[n] - p0[n]) * 1000) for n in free}
    drift = {n: float(np.linalg.norm(p2[n] - p1[n]) * 1000) for n in free}
    rep["physics"] = {"settle_mm": settle, "drift_mm": drift, "worst_contact": worst}
    errs = []
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
    rep.update(stage="physics", errors=errs, ok=not errs)
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
    print("CHECK", "PASS" if rep["ok"] else f"FAIL[{rep['stage']}]", json.dumps(rep["errors"])[:400])


if __name__ == "__main__":
    main()
