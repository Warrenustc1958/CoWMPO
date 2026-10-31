"""Target-private diagnostic: GT rack AABB from COLLISION geoms only (contype != 0, same as the v8 scale-scan
diagnostic) vs from all geoms, for the 60 dev32+fit28 episodes, and the frozen v31c18 / world11 Builder records scored
against both. The original s5_gt metric is kept unchanged; this only tests whether the constant S5 offset is a
geometry-definition mismatch. Results stay in target_eval."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020/rack_s5_static_v33")
T = Path("/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/rack_s5_static_v33")
INITS = {"dev": Path("/data_cf/cowmpo/init/libero_goal_dev3"), "fit": Path("/data_cf/cowmpo/init/libero_goal_fit3")}


def gt(args):
    role, ep = args
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from cowm.init.aabb import object_aabb
    I = INITS[role] if (INITS[role] / "manifest.json").exists() else INITS["dev"]
    man = json.loads((I / "manifest.json").read_text())
    if ep + ".npy" not in man["episodes"]:
        I = INITS["fit" if I == INITS["dev"] else "dev"]
        man = json.loads((I / "manifest.json").read_text())
    info = man["episodes"][ep + ".npy"]
    suite = benchmark.get_benchmark_dict()[man["suite"]]()
    t = suite.get_task(info["task_id"])
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                             camera_heights=64, camera_widths=64)
    env.reset()
    pv = np.load(I / info["privileged"])
    m, d = env.env.sim.model, env.env.sim.data
    m.body_pos[:] = pv["body_pos"]
    m.body_quat[:] = pv["body_quat"]
    env.set_init_state(suite.get_task_init_states(info["task_id"])[info["trial_id"]])
    w = np.zeros(7)
    w[6] = -1
    for _ in range(man["num_steps_wait"]):
        env.step(w)
    root = int(m.body_rootid[m.body_name2id("wine_rack_1_main")])
    allg = [g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == root]
    col = [g for g in allg if m.geom_contype[g] or m.geom_conaffinity[g]]
    base = d.body_xpos[m.body_name2id(env.env.robots[0].robot_model.root_body)].copy()
    out = {}
    for name, gs in (("all", allg), ("collision", col)):
        c, h = object_aabb(m, d, gs)
        out[name] = {"center_rel_base": (c - base).tolist(), "half": np.asarray(h).tolist(), "n_geoms": len(gs)}
    env.close()
    return ep, out


def main():
    T.mkdir(parents=True, exist_ok=True)
    jobs = []
    for role in ("dev", "fit"):
        for row in json.loads((R / "v31c18" / f"{role}.json").read_text())["rows"]:
            jobs.append((role, row["episode"]))
    gp = T / "gt_rack_aabb_all_vs_collision.json"
    if gp.exists():
        G = json.loads(gp.read_text())
    else:
        with ProcessPoolExecutor(20) as pool:
            G = dict(pool.map(gt, jobs))
    (T / "gt_rack_aabb_all_vs_collision.json").write_text(json.dumps(G, indent=1))
    rep = {}
    for model in ("world11", "v31c18", "v36c03"):
        for role in ("dev", "fit"):
            rows = json.loads((R / model / f"{role}.json").read_text())["rows"]
            for kind in ("all", "collision"):
                e, dx, dy, ext = [], [], [], []
                for r in rows:
                    g = G[r["episode"]][kind]
                    dd = np.asarray(r["aabb_center_rel_base"]) - g["center_rel_base"]
                    e.append(float(np.linalg.norm(dd[:2]) * 100)); dx.append(dd[0] * 100); dy.append(dd[1] * 100)
                    ext.append(float(np.max(np.abs(np.asarray(r["aabb_half"]) - g["half"])) * 200))
                rep[f"{model}|{role}|gt_{kind}"] = {
                    "n": len(e), "within_3cm": int(sum(x < 3 for x in e)), "xy_mean_cm": round(float(np.mean(e)), 3),
                    "xy_p90_cm": round(float(np.percentile(e, 90)), 3), "dx_mean": round(float(np.mean(dx)), 3),
                    "dy_mean": round(float(np.mean(dy)), 3), "extent_err_cm": round(float(np.mean(ext)), 3),
                    "gt_full_extent_cm": (np.asarray(G[rows[0]["episode"]][kind]["half"]) * 200).round(1).tolist(),
                    "model_full_extent_cm": (np.asarray(rows[0]["aabb_half"]) * 200).round(1).tolist()}
    (T / "summary_collision_vs_all.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep.items():
        print("RACK_S5_COL", k, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
