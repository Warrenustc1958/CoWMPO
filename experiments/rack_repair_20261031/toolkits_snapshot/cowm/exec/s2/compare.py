"""S2 gate: compare base-relative EE pose, joints and gripper between target and mirror."""
import argparse
import json

import numpy as np

from actions import make_sequences


def quat_angle(q1, q2):
    dot = np.abs(np.sum(q1 * q2, axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(dot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--mirror", required=True)
    p.add_argument("--pos-tol-mm", type=float, default=2.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t, m = np.load(args.target), np.load(args.mirror)

    rows = {}
    for name in make_sequences():
        tp = t[f"{name}/eef_pos"] - t[f"{name}/base_pos"]
        mp = m[f"{name}/eef_pos"] - m[f"{name}/base_pos"]
        pos_err = np.linalg.norm(tp - mp, axis=-1) * 1000.0
        rot_err = np.degrees(quat_angle(t[f"{name}/eef_quat"], m[f"{name}/eef_quat"]))
        rows[name] = {
            "pos_err_mm_max": float(pos_err.max()),
            "pos_err_mm_final": float(pos_err[-1]),
            "rot_err_deg_max": float(rot_err.max()),
            "joint_err_rad_max": float(np.abs(t[f"{name}/joint_pos"] - m[f"{name}/joint_pos"]).max()),
            "grip_err_max": float(np.abs(t[f"{name}/gripper_qpos"] - m[f"{name}/gripper_qpos"]).max()),
            "target_travel_mm": float(np.linalg.norm(tp[-1] - tp[0]) * 1000.0),
            "mirror_travel_mm": float(np.linalg.norm(mp[-1] - mp[0]) * 1000.0),
            "target_init_qvel_norm": float(t[f"{name}/init_qvel_norm"]) if f"{name}/init_qvel_norm" in t else -1.0,
        }
    worst = max(r["pos_err_mm_max"] for r in rows.values())
    summary = {"pos_tol_mm": args.pos_tol_mm, "worst_pos_err_mm": worst,
               "pass": bool(worst <= args.pos_tol_mm), "per_sequence": rows}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{'seq':18s} {'pos_max':>8s} {'pos_end':>8s} {'rot_max':>8s} {'jnt_max':>8s} {'grip':>7s} "
          f"{'t_trav':>7s} {'m_trav':>7s} {'qvel0':>7s}")
    for k, r in rows.items():
        print(f"{k:18s} {r['pos_err_mm_max']:8.3f} {r['pos_err_mm_final']:8.3f} {r['rot_err_deg_max']:8.3f} "
              f"{r['joint_err_rad_max']:8.4f} {r['grip_err_max']:7.4f} {r['target_travel_mm']:7.1f} "
              f"{r['mirror_travel_mm']:7.1f} {r['target_init_qvel_norm']:7.4f}")
    print("WORST pos err mm:", round(worst, 3), "PASS" if summary["pass"] else "FAIL")


if __name__ == "__main__":
    main()
