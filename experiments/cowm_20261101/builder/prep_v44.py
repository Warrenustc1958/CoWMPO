import json
from pathlib import Path
from cowm.world.artifact import tree_sha
R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
H = R / "rack_hfield_v43"
rep = json.loads((H / "report.json").read_text())
it = rep["history"][rep["selected"]]
assert it["iter"] == 3
w = Path(it["world"])
base = json.loads((R / "rack_far_dev_v36/cand03/result.json").read_text())
(w.parent / "result.json").write_text(json.dumps({"world": str(w), "world_tree_sha256": tree_sha(w),
    "states": base["states"], "source": str(H / "report.json"), "iter": 3}, indent=1))
sel = {"cand": "rack_hfield_v43__iter3", **{k: it[k] for k in ("trans_err_mean", "trans_err_max", "n_ge3", "n", "public_static_iou")},
       "loo_last_layer": rep["history"][2]["layer"]}
(H / "score_v44.json").write_text(json.dumps({"candidates": [sel], "GT_inputs": False}, indent=1))
print("PREP_OK", tree_sha(w)[:12])
