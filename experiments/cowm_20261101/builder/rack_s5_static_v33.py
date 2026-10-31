# Copyright 2026 The RLinf Authors.
"""Builder side (no GT): rack static S5 record for dev32 + fit28 under the current natural object and RGB robot
initialization (same sources as the Codex stove revalidation, builder_batch.py), for the frozen rack v31 cand18 and,
for reference, the unrepaired combined world11. Per episode: rack AABB centre relative to the robot base, AABB half
size, rendered rack mask. Scored against GT only by the Target script target_rack_s5_static_v33.py.
usage: python rack_s5_static_v33.py <role> <model_name>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from cowm import roles
from cowm.init.aabb import object_aabb
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
OUT = R / "rack_s5_static_v33"
MODELS = {"v31c18": R / "rack_rib_dev_v31/cand18/world", "world11": R / "bowl_support_dev_v1/world11", "v36c03": R / "rack_far_dev_v36/cand03/world", "v48i5": R / "rack_hfield_v48/iter5/world"}
DIRS = {"dev": ("natural_states_all_dev_v1", "robot_rgb_all_dev_v2"), "fit": ("natural_bowl_fit_v2", "robot_rgb_fit_v2")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(role, name):
    world = MODELS[name]
    sd, rd = DIRS[role]
    eps = sorted(p.parent.name for p in (R / sd).glob("*/state.json"))
    roles.require(eps, {role}, "rack static S5 revalidation")
    if len(eps) != {"dev": 32, "fit": 28}[role]:
        raise ValueError("unexpected role population")
    scene = Scene(str(world))
    k = scene.names.index("rack")
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in eps:
        st, rb = R / sd / ep / "state.json", R / rd / ep / "robot.json"
        scene.reset()
        apply_state(scene, json.loads(st.read_text())["objects"], require_matched=True,
                    robot=json.loads(rb.read_text())["robot"])
        c, h = object_aabb(scene.env.sim.model, scene.env.sim.data, np.flatnonzero(scene.geom2obj == k))
        np.savez_compressed(out / f"{ep}.npz", rack=scene.seg() == k)
        rows.append({"episode": ep, "aabb_center_rel_base": (c - scene.base).tolist(), "aabb_half": h.tolist(),
                     "state_sha256": sha(st), "robot_sha256": sha(rb)})
    scene.env.close()
    (out / f"{role}.json").write_text(json.dumps({"role": role, "model": name, "world": str(world),
                                                  "world_tree_sha256": tree_sha(world), "count": len(rows),
                                                  "GT_inputs": False, "rows": rows}, indent=1))
    print("RACK_S5_BUILDER_DONE", role, name, len(rows), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
