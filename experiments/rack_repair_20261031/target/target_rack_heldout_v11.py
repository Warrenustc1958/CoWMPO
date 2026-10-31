"""Held-out functional acceptance of the frozen rack model (Target side; results stay in target_eval).

Probes: 5 frozen vertical actions never used for model selection. Rule fixed before looking at results:
  valid   : Target contact phase has gripper-rack contact, no arm contact, no other-object contact (judged on
            Target contact types only, before any model comparison; invalid probes are reported, not dropped)
  pass    : |first contact step (model) - (Target)| <= 5 AND |eef (base frame) at first contact| diff <= 2 cm
Model: frozen v8 scale00 (rack_scale_scan_dev_v8/scale00), unchanged.
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

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
P = R / "rack_vertical_preflight_dev_v3"
T = Path("/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/rack_heldout_v11")
INIT = Path("/data_cf/cowmpo/init/libero_goal_dev3")
EP = "libero_goal_t0_i0"
KEYS = ["vert_x-080_y-080", "vert_x-080_y+000", "vert_x+080_y-080", "vert_x+080_y+000", "vert_x+080_y+080"]
MODEL = R / "rack_scale_scan_dev_v8/scale00"
TOL_STEPS, TOL_CM = 5, 2.0


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
        grip, arm, other = set(), set(), set()
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


def model_run(probe, robot_rec):
    res = json.loads((MODEL / "result.json").read_text())
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


def main():
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
               "target_other_contact_steps": sum(bool(set(r["grip"]) - {"rack"}) for r in rows)}
        if valid:
            mf = model_run(probe, robot_rec)
            rec["model_first_step"] = mf[0] if mf else None
            if mf:
                rec["step_err"] = mf[0] - t_first["step"]
                rec["eef_err_cm"] = round(float(np.linalg.norm(np.asarray(mf[1]) - t_first["eef_base"]) * 100), 2)
                rec["pass"] = abs(rec["step_err"]) <= TOL_STEPS and rec["eef_err_cm"] <= TOL_CM
            else:
                rec["pass"] = False
        out.append(rec)
        print(json.dumps(rec), flush=True)
    v = [r for r in out if r["valid"]]
    summ = {"rule": {"tol_steps": TOL_STEPS, "tol_cm": TOL_CM}, "model": str(MODEL), "probes": out,
            "valid": len(v), "passed": sum(r.get("pass", False) for r in v)}
    (T / "summary.json").write_text(json.dumps(summ, indent=1))
    print("HELDOUT_SUMMARY valid", summ["valid"], "passed", summ["passed"], "of", len(KEYS))


if __name__ == "__main__":
    main()
