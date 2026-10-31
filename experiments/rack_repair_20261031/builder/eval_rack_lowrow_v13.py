# Copyright 2026 The RLinf Authors.
"""Is a contact-position-sensitive, Builder-visible signal able to rank the v12 rack candidates?

Signal: per frame, the LOWEST image row of the robot mask (a proxy for gripper height when it pushes down),
observed = frozen v7b public tracks, predicted = the candidate world's rendered robot mask. Scored on ALL
saved frames from 20 steps before the contact phase to its end (descent + contact), 9 probes. No Target data.
usage: python eval_rack_lowrow_v13.py
"""
import json
from pathlib import Path

import numpy as np

from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
V12 = R / "rack_back_dev_v12"
EP = "libero_goal_t0_i0"


def lowrow(m):
    ys = np.nonzero(m.any(axis=1))[0]
    return int(ys.max()) if len(ys) else None


def main():
    qs = json.loads((V12 / "freeze.json").read_text())["queries"]
    robot0 = json.loads((R / "robot_rgb_all_dev_v2" / EP / "robot.json").read_text())["robot"]
    rows = []
    for cand in sorted(V12.glob("cand*/result.json")):
        res = json.loads(cand.read_text())
        sc = Scene(res["world"])
        doc = json.loads((Path(res["states"]) / EP / "state.json").read_text())
        errs, per = [], {}
        for q in qs:
            probe = json.loads(Path(q["source"]).read_text())
            ph = list(probe["phases"])
            i = ph.index("contact")
            start = probe["phases"][ph[i - 1]] - 20
            end = probe["phases"]["contact"]
            frames = [s for s in probe["frames"] if start < s <= end]
            sc.reset()
            apply_state(sc, doc["objects"], require_matched=True, robot=robot0)
            e = []
            for step, act in enumerate(probe["actions"], 1):
                sc.env.step(np.asarray(act))
                if step in frames:
                    with np.load(Path(q["video"]) / f"{EP}__f{step:03d}" / "masks.npz") as a:
                        obs = lowrow(a["robot"])
                    pred = lowrow(sc.robot_mask())
                    if obs is not None and pred is not None:
                        e.append(abs(pred - obs))
                if step >= end:
                    break
            per[q["key"]] = round(float(np.mean(e)), 2) if e else None
            errs += e
        sc.env.close()
        rows.append({"cand": cand.parent.name, "e": res["e"], "h": res["h"], "builder_loss": round(res["loss"], 4),
                     "lowrow_err_px_mean": round(float(np.mean(errs)), 3), "per_probe": per})
        print(json.dumps(rows[-1]), flush=True)
    (V12 / "lowrow_v13.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
