"""Target-private diagnostic for the rack size scan v8: collision-only 3D extent and xy error per scale (6 dev).
GT only scores; nothing goes back to the Builder."""
import json
import os
from pathlib import Path

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from cowm.init.aabb import object_aabb
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
T = Path("/data_cf/cowmpo/target_eval/codex_s56_20261030_1020")
SCAN = R / "rack_lattice_dev_v10"
I = Path("/data_cf/cowmpo/init/libero_goal_dev3")
man = json.loads((I / "manifest.json").read_text())["episodes"]
suite = benchmark.get_benchmark_dict()["libero_goal"]()
EPS = [f"libero_goal_t{k}_i0" for k in range(6)]

gt = {}
for ep in EPS:
    info = man[ep + ".npy"]
    t = suite.get_task(info["task_id"])
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                             camera_heights=64, camera_widths=64)
    env.reset()
    pv = np.load(I / info["privileged"])
    m, d = env.env.sim.model, env.env.sim.data
    m.body_pos[:] = pv["body_pos"]
    m.body_quat[:] = pv["body_quat"]
    env.set_init_state(suite.get_task_init_states(info["task_id"])[info["trial_id"]])
    for _ in range(15):
        env.step(np.zeros(7))
    root = m.body_name2id("wine_rack_1_main")
    gs = [g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == int(m.body_rootid[root]) and m.geom_contype[g]]
    c, h = object_aabb(m, d, gs)
    gt[ep] = (c - d.body_xpos[m.body_name2id(env.env.robots[0].robot_model.root_body)], 2 * np.asarray(h))
    env.close()

rows = []
for res in sorted(SCAN.glob("candidate*/result.json")):
    r = json.loads(res.read_text())
    sc = Scene(r["world"])
    ms, ds = sc.env.sim.model, sc.env.sim.data
    k = sc.names.index("rack")
    errs, ext = [], None
    for ep in EPS:
        sc.reset()
        apply_state(sc, json.loads((Path(r["states"]) / ep / "state.json").read_text())["objects"])
        mg = [g for g in range(ms.ngeom) if int(sc.geom2obj[g]) == k and ms.geom_contype[g]]
        c, h = object_aabb(ms, ds, mg)
        base = ds.body_xpos[ms.body_name2id(sc.env.robots[0].robot_model.root_body)]
        errs.append(float(np.linalg.norm(((c - base) - gt[ep][0])[:2]) * 100))
        ext = 2 * np.asarray(h) * 100
    sc.env.close()
    rows.append({"seed": r.get("seed"), "dims_m": [round(v, 3) for v in r["dims_m"]], "builder_loss": round(r["loss"], 4), "static_iou": round(r["static_iou"], 4),
                 "robot_iou": round(r["robot_iou"], 4), "silhouette_mse": round(r["silhouette_mse"], 5),
                 "model_extent_cm": np.round(ext, 1).tolist(), "gt_extent_cm": np.round(gt[EPS[0]][1] * 100, 1).tolist(),
                 "xy_err_cm_mean": round(float(np.mean(errs)), 2), "within_3cm": sum(e < 3 for e in errs)})
(T / "rack_lattice_v10_diagnostic.json").write_text(json.dumps(rows, indent=1))
for row in rows:
    print(json.dumps(row))
