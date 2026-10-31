"""Target side: extract the agentview camera of the target platform as a DECLARED shared prior (plan §3.7A).

Exports the camera pose RELATIVE TO THE ROBOT BASE (so it transfers to any table layout), fovy and the image
convention. Also reports the table-top height relative to the robot base. Checks all tasks of the suite give the
same camera. No object, fixture or region information is exported.
"""
import argparse
import json
import os

import numpy as np


def quat_wxyz_from_mat(R):
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()[a.suite]()
    rows = []
    for t in range(suite.n_tasks):
        task = suite.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_offscreen_renderer=False,
                         has_renderer=False, control_freq=20)
        env.reset()
        sim = env.env.sim
        m, d = sim.model, sim.data
        cid = m.camera_name2id("agentview")
        bid = m.body_name2id(env.env.robots[0].robot_model.root_body)
        Rb = d.body_xmat[bid].reshape(3, 3)
        pb = d.body_xpos[bid].copy()
        Rc = d.cam_xmat[cid].reshape(3, 3)
        pc = d.cam_xpos[cid].copy()
        rel_pos = Rb.T @ (pc - pb)
        rel_R = Rb.T @ Rc
        table_top = env.env.table_offset[2] if hasattr(env.env, "table_offset") else None
        rows.append({"task": t, "rel_pos": rel_pos.tolist(), "rel_quat": quat_wxyz_from_mat(rel_R).tolist(),
                     "fovy": float(m.cam_fovy[cid]), "base_pos": pb.tolist(),
                     "table_top_z_rel_base": None if table_top is None else float(table_top - pb[2])})
        env.close()
    p0 = np.array(rows[0]["rel_pos"])
    spread = max(float(np.abs(np.array(r["rel_pos"]) - p0).max()) for r in rows)
    prior = {"source": f"target platform agentview camera ({a.suite}), relative to robot base; declared shared prior",
             "rel_pos": rows[0]["rel_pos"], "rel_quat": rows[0]["rel_quat"], "fovy": rows[0]["fovy"],
             "image_size": [256, 256], "image_flip": "rotate 180 deg (img[::-1, ::-1]) as in RLinf LIBERO",
             "max_rel_pos_spread_across_tasks_m": spread,
             "table_top_z_rel_base": rows[0]["table_top_z_rel_base"], "per_task": rows}
    with open(a.out, "w") as f:
        json.dump(prior, f, indent=2)
    print(json.dumps({k: prior[k] for k in ("rel_pos", "rel_quat", "fovy", "max_rel_pos_spread_across_tasks_m",
                                            "table_top_z_rel_base")}))


if __name__ == "__main__":
    main()
