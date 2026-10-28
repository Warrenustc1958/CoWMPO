"""Contact rule test (Builder side, simulation only). Exit code 1 on failure.

legacy (default)  : world.json has no contact_rule, no <pair>s, no geom priority changes.
object_table_pair : bowl-table contact friction equals the ceramic material value (runtime override via
                    apply_params), free slide distance decreases monotonically with it, no geom priority changes,
                    and robot/fixture-table contacts keep MuJoCo's default mixing (no pair involves them).
"""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

from cowm.init.estimate_state import Scene
from cowm.s6.params import apply_params
from cowm.world.compile_world import compile_world

PROG = "/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/world/ref_goal_program_v03.py"


def table_contacts(scene, obj):
    m, d = scene.env.sim.model, scene.env.sim.data
    k = scene.names.index(obj)
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        g = [int(c.geom1), int(c.geom2)]
        if any(scene.geom2obj[x] == k for x in g) and any((m.geom_id2name(x) or "").startswith("table_collision")
                                                           for x in g):
            out.append(float(c.friction[0]))
    return out


def slide(scene, mu, v0=0.3, steps=60):
    apply_params(scene, {"ceramic.friction": mu})
    scene.reset()
    sim = scene.env.sim
    for k, (o, ob) in enumerate(zip(scene.world["objects"], scene.env.objects)):   # clear the table
        if o["name"] == "bowl":
            continue
        if o["free"]:
            sim.data.set_joint_qpos(ob.joints[0], np.array([-1.5, 0.3 * k, 0.5, 1, 0, 0, 0]))
        else:
            sim.model.body_pos[sim.model.body_name2id(ob.root_body)] = [-1.5, 0.3 * k, 0.5]
    jid = sim.model.joint_name2id(scene.env.obj("bowl").joints[0])
    qp, dv = sim.model.jnt_qposadr[jid], sim.model.jnt_dofadr[jid]
    sim.data.qpos[qp:qp + 7] = [0.05, 0.10, scene.world["table"]["height"] + 0.003, 1, 0, 0, 0]
    sim.forward()
    for _ in range(100):
        sim.step()
    fr = table_contacts(scene, "bowl")
    sim.data.qvel[dv:dv + 6] = [0, -v0, 0, 0, 0, 0]
    p0 = scene.env.object_pos("bowl").copy()
    for i in range(steps * 25):
        sim.step()
        if i == 20:
            fr += table_contacts(scene, "bowl")
    return fr, float(np.linalg.norm(scene.env.object_pos("bowl")[:2] - p0[:2]))


def main():
    out = sys.argv[1]
    rep, ok = {}, True
    # legacy: unchanged semantics
    wl = os.path.join(out, "world_legacy")
    compile_world(PROG, wl)
    jl = json.load(open(os.path.join(wl, "world.json")))
    sl = Scene(wl)
    ml = sl.env.sim.model
    rep["legacy"] = {"contact_rule_key": "contact_rule" in jl["table"], "npair": int(ml.npair),
                     "max_priority": int(np.abs(ml.geom_priority).max())}
    ok &= not rep["legacy"]["contact_rule_key"] and rep["legacy"]["npair"] == 0 and rep["legacy"]["max_priority"] == 0
    sl.env.close()
    # opt-in pair rule
    wp = os.path.join(out, "world_pair")
    compile_world(PROG, wp, contact_rule="object_table_pair")
    jp = json.load(open(os.path.join(wp, "world.json")))
    sp = Scene(wp)
    mp = sp.env.sim.model
    pair_objs = sorted({sp.names[int(sp.geom2obj[g])] for p in range(mp.npair)
                        for g in (int(mp.pair_geom1[p]), int(mp.pair_geom2[p])) if sp.geom2obj[g] >= 0})
    free = sorted(o["name"] for o in jp["objects"] if o["free"])
    rep["pair"] = {"contact_rule": jp["table"].get("contact_rule"), "npair": int(mp.npair),
                   "pair_objects": pair_objs, "free_objects": free,
                   "max_priority": int(np.abs(mp.geom_priority).max()), "runs": []}
    ok &= rep["pair"]["contact_rule"] == "object_table_pair" and pair_objs == free and rep["pair"]["max_priority"] == 0
    lo, hi = jp["params"]["ceramic.friction"]["bounds"]
    for mu in [lo, 0.7, hi]:
        fr, d = slide(sp, mu)
        rep["pair"]["runs"].append({"ceramic_friction": mu, "contact_friction": sorted(set(round(f, 4) for f in fr)),
                                    "slide_cm": round(d * 100, 3)})
        print(rep["pair"]["runs"][-1], flush=True)
    runs = rep["pair"]["runs"]
    ok &= all(r["contact_friction"] and all(abs(f - r["ceramic_friction"]) < 1e-6 for f in r["contact_friction"])
              for r in runs)
    ok &= runs[0]["slide_cm"] > runs[1]["slide_cm"] > runs[2]["slide_cm"]
    sp.env.close()
    rep["ok"] = bool(ok)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    print("legacy", rep["legacy"], "pair", {k: v for k, v in rep["pair"].items() if k != "runs"})
    print("CONTACT_RULE_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
