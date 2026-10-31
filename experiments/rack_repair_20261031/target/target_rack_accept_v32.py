"""Independent functional acceptance of the panel-repaired rack model (v32; Target side, results stay in target_eval).

Probes: the 6 positions of rack_accept_plan_v32 (x+040/x+060/x+110 y-060, x+050/x+090 y-040, x+070_y+040), never used for planning, fitting
or any Target run. The rule is read from the plan freeze (fixed before execution, identical to v22):
  valid  : Target contact phase has gripper-rack contact, no arm contact, no other-object contact, no pre-contact
           (judged on Target contact types only, before any model comparison; invalid probes are reported)
  pass   : |first contact step (model) - (Target)| <= 5 AND |eef (base frame) at first contact| diff <= 2 cm
  set    : ACCEPT iff >= 4 valid and every valid probe passes
Models: frozen v31 selection (decision); v29 cand10, v25 cand01, v23 cand03, v21 cand07, unrepaired v8 (reference only).
Derived from target_rack_accept_v22.py (same Target run and first-contact definitions).
"""
import json
import os
from pathlib import Path

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from cowm.init.estimate_state import Scene
from cowm.init.gt_dump import NAME_MAP
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import verify_tree

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
P = R / "rack_accept_plan_v32"
T = Path("/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/rack_accept_v32")
INIT = Path("/data_cf/cowmpo/init/libero_goal_dev3")
EP = "libero_goal_t0_i0"
FREEZE = json.loads((P / "freeze.json").read_text())
KEYS = sorted(FREEZE["grid_mm"])
RULE = FREEZE["rule"]
MODELS = {"frozen": Path(FREEZE["frozen_model"]["result"]).parent,
          **{k: Path(p) for k, p in FREEZE["reference_models"].items()}}


def target_run(probe):
    man = json.loads((INIT / "manifest.json").read_text())
    info = man["episodes"][EP + ".npy"]
    suite = benchmark.get_benchmark_dict()[man["suite"]]()
    task = suite.get_task(info["task_id"])
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                             camera_heights=64, camera_widths=64, ignore_done=True)
    env.reset()
    sim = env.env.sim
    m = sim.model
    pv = np.load(INIT / info["privileged"])
    m.body_pos[:] = pv["body_pos"]
    m.body_quat[:] = pv["body_quat"]
    env.set_init_state(suite.get_task_init_states(info["task_id"])[info["trial_id"]])
    warm = np.zeros(7)
    warm[6] = -1
    for _ in range(man["num_steps_wait"]):
        env.step(warm)
    rroot = m.body_name2id(env.env.robots[0].robot_model.root_body)
    robot = np.array([int(m.body_rootid[m.geom_bodyid[g]]) == rroot or
                      (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith(("robot0", "gripper0")) for g in range(m.ngeom)])
    names = {b: n for n, pre in NAME_MAP.items() for b in range(m.nbody)
             if int(m.body_rootid[b]) == b and (m.body_id2name(b) or "").startswith(pre + "_")}
    base = sim.data.body_xpos[rroot].copy()
    rows = []
    for step, act in enumerate(probe["actions"], 1):
        obs, _, _, _ = env.step(np.asarray(act))
        grip, arm = set(), set()
        for c in sim.data.contact[:sim.data.ncon]:
            for rg, og in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                if robot[rg] and not robot[og]:
                    n = names.get(int(m.body_rootid[m.geom_bodyid[og]]), "environment")
                    (grip if (m.body_id2name(int(m.geom_bodyid[rg])) or "").startswith("gripper0") else arm).add(n)
        phase = next(p for p, last in probe["phases"].items() if step <= last)
        rows.append({"step": step, "phase": phase, "grip": sorted(grip), "arm": sorted(arm),
                     "eef_base": (np.asarray(obs["robot0_eef_pos"]) - base).tolist()})
    env.close()
    return rows


def model_run(probe, robot_rec, model):
    res = json.loads((model / "result.json").read_text())
    sc = Scene(res["world"])
    m = sc.env.sim.model
    k = sc.names.index("rack")
    grip = np.array([(m.body_id2name(int(m.geom_bodyid[g])) or "").startswith("gripper0") for g in range(m.ngeom)])
    sc.reset()
    apply_state(sc, json.loads((Path(res["states"]) / EP / "state.json").read_text())["objects"],
                require_matched=True, robot=robot_rec)
    base = sc.env.sim.data.body_xpos[m.body_name2id(sc.env.robots[0].robot_model.root_body)].copy()
    first = None
    for step, act in enumerate(probe["actions"], 1):
        sc.env.step(np.asarray(act))
        d = sc.env.sim.data
        if any((grip[int(c.geom1)] and sc.geom2obj[int(c.geom2)] == k) or (grip[int(c.geom2)] and sc.geom2obj[int(c.geom1)] == k)
               for c in d.contact[:d.ncon]):
            first = (step, (sc.env.eef_pos() - base).tolist())
            break
    sc.env.close()
    return first


def compare(mf, t_first):
    if not mf:
        return {"model_first_step": None, "pass": False}
    rec = {"model_first_step": mf[0], "step_err": mf[0] - t_first["step"],
           "eef_err_cm": round(float(np.linalg.norm(np.asarray(mf[1]) - t_first["eef_base"]) * 100), 2)}
    rec["pass"] = abs(rec["step_err"]) <= RULE["tol_steps"] and rec["eef_err_cm"] <= RULE["tol_cm"]
    return rec


def main():
    fz = FREEZE["frozen_model"]
    verify_tree(fz["world"], fz["world_tree_sha256"])
    T.mkdir(parents=True, exist_ok=False)
    robot_rec = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    out = []
    for key in KEYS:
        probe = json.loads((P / key / "probe.json").read_text())
        rows = target_run(probe)
        (T / f"{key}_target_rows.json").write_text(json.dumps(rows))
        cp = [r for r in rows if r["phase"] == "contact"]
        pre = [r for r in rows if r["phase"] != "contact" and (r["grip"] or r["arm"])]
        valid = (any("rack" in r["grip"] for r in cp) and not any(r["arm"] for r in rows)
                 and not any(set(r["grip"]) - {"rack"} for r in rows) and not pre)
        t_first = next((r for r in rows if "rack" in r["grip"]), None)
        rec = {"key": key, "valid": valid, "target_first_step": t_first["step"] if t_first else None,
               "target_arm_contact_steps": sum(bool(r["arm"]) for r in rows),
               "target_other_contact_steps": sum(bool(set(r["grip"]) - {"rack"}) for r in rows),
               "target_precontact_steps": len(pre)}
        if valid:
            for name, model in MODELS.items():
                rec[name] = compare(model_run(probe, robot_rec, model), t_first)
        out.append(rec)
        print(json.dumps(rec), flush=True)
    v = [r for r in out if r["valid"]]
    passed = {name: sum(r[name]["pass"] for r in v) for name in MODELS}
    summ = {"rule": RULE, "models": {k: str(p) for k, p in MODELS.items()}, "probes": out, "valid": len(v),
            "passed": passed,
            "decision": "ACCEPT" if len(v) >= RULE["min_valid"] and passed["frozen"] == len(v) else "REJECT"}
    (T / "summary.json").write_text(json.dumps(summ, indent=1))
    print("ACCEPT_V32_SUMMARY valid", len(v), "passed", passed, "decision", summ["decision"], flush=True)


if __name__ == "__main__":
    main()
