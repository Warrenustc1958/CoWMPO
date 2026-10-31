"""Target-private: score the Builder rack static records (rack_s5_static_v33) against s5_gt, same metric as the
Codex stove summary (AABB centre XY error rel. robot base, <3 cm, P90, extent error, render/GT mask IoU).
Results stay in target_eval; nothing goes back to the Builder."""
import json
from pathlib import Path

import numpy as np

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020/rack_s5_static_v33")
G = Path("/data_cf/cowmpo/target_eval/s5_gt")
T = Path("/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/rack_s5_static_v33")


def main():
    T.mkdir(parents=True, exist_ok=True)
    report = {}
    for model in ("world11", "v31c18"):
        report[model] = {}
        for role in ("dev", "fit"):
            data = json.loads((R / model / f"{role}.json").read_text())
            rows = []
            for row in data["rows"]:
                ep = row["episode"]
                root = G / ("dev3" if int(ep.split("_i")[1]) < 3 else "fit3")
                gt = json.loads((root / f"{ep}.json").read_text())["rack"]
                d = np.asarray(row["aabb_center_rel_base"]) - gt["aabb_center_rel_base"]
                with np.load(root / f"{ep}_masks.npz") as a, np.load(R / model / f"{ep}.npz") as b:
                    x, y = a["rack"], b["rack"]
                rows.append({"episode": ep, "xy_cm": float(np.linalg.norm(d[:2]) * 100), "dx_cm": float(d[0] * 100),
                             "dy_cm": float(d[1] * 100), "z_cm": float(abs(d[2]) * 100),
                             "extent_cm": float(np.max(np.abs(np.asarray(row["aabb_half"]) - gt["aabb_half"])) * 200),
                             "render_gt_iou": float((x & y).sum() / max(1, (x | y).sum()))})
            report[model][role] = {"n": len(rows), "within_3cm": sum(r["xy_cm"] < 3 for r in rows),
                                   "means": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "episode"},
                                   "xy_p90_cm": float(np.percentile([r["xy_cm"] for r in rows], 90)),
                                   "model_full_extent_cm": (np.asarray(data["rows"][0]["aabb_half"]) * 200).round(1).tolist(),
                                   "gt_full_extent_cm": (np.asarray(gt["aabb_half"]) * 200).round(1).tolist()}
            (T / f"{model}_{role}_rows.json").write_text(json.dumps(rows, indent=1))
    (T / "summary.json").write_text(json.dumps(report, indent=1))
    for m, v in report.items():
        for r, s in v.items():
            print("RACK_S5", m, r, json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
