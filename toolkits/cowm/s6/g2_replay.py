"""S6 G2 target-evidence gate + replay metrics (Builder side, plan §9.7-D).

Replays the probe actions the Target executed in the code world (S5 state applied) and compares the
world's segmentation render with SAM2 masks of the returned probe frames:
  static   = mean IoU over observed objects at step 0
  dynamic  = probe-object mask-centroid error (px) over the probe frames
  terminal = probe-object IoU at the last probe frame
Evidence validity is decided ONCE, before any candidate is scored, and frozen in a manifest
(build_validity): a candidate world never decides which evidence frames are excluded. A frame that is valid
evidence but where the candidate does not render the probe object counts as a failure, not as a skip.

G2 passes iff (thresholds fixed before fitting; audit 10.28 added terminal + coverage):
  static_iou >= G2_STATIC_IOU, dyn_err_px <= G2_DYN_PX, final_obj_iou >= G2_FINAL_IOU,
  coverage >= G2_MIN_COVERAGE  (valid frames actually measured / expected probe frames), and no episode whose
  dynamic error is missing entirely.
Runtime parameter overrides: <material>.friction (geom friction of objects with that material).
"""
import argparse
import json
import os

import numpy as np

from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state

G2_STATIC_IOU = 0.6
G2_DYN_PX = 6.0
G2_FINAL_IOU = 0.5        # added 10.28, before the formal gate split is used
G2_MIN_COVERAGE = 0.9     # added 10.28
ROBOT_FRAC = 0.5          # evidence mask mostly covered by the robot -> perception error (same rule as S5)


def set_friction(scene, material, value):
    m = scene.env.sim.model
    for k, o in enumerate(scene.world["objects"]):
        if o.get("material") == material:
            m.geom_friction[scene.geom2obj == k, 0] = value


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def mask_path(masks_root, ep, step):
    return os.path.join(masks_root, f"{ep}__f{step:03d}", "masks.npz")


def default_validity(masks_root):
    """Frozen manifest lives next to the probe masks it describes (one per evidence set)."""
    return os.path.normpath(masks_root) + "_validity.json"


def build_validity(ref_world, state_root, probe_root, masks_root, episodes, out):
    """Frozen evidence manifest. Per expected frame and object: valid | missing_file | empty | robot_occluded.
    The robot mask comes from ONE designated reference world (the robot is the shared platform prior and is
    driven by the same actions), so candidate worlds cannot change the evidence set."""
    scene = Scene(ref_world)
    man = {"ref_world": ref_world, "robot_frac": ROBOT_FRAC, "episodes": {}}
    try:
        for ep in episodes:
            state = json.load(open(os.path.join(state_root, ep, "state.json")))["objects"]
            probe = json.load(open(os.path.join(probe_root, ep, "probe.json")))
            frames = sorted(set(probe["frames"]) | {0})
            scene.reset()
            apply_state(scene, state)
            rec = {}

            def check(step):
                f = mask_path(masks_root, ep, step)
                if not os.path.isfile(f):
                    rec[str(step)] = {"_frame": "missing_file"}
                    return
                mk, rob = np.load(f), scene.robot_mask()
                st = {}
                for n in mk.files:
                    s = int(mk[n].sum())
                    st[n] = "empty" if s == 0 else ("robot_occluded" if (mk[n] & rob).sum() / s > ROBOT_FRAC
                                                    else "valid")
                rec[str(step)] = st

            if 0 in frames:
                check(0)
            for k, act in enumerate(probe["actions"], start=1):
                scene.env.step(np.asarray(act))
                if k in frames:
                    check(k)
            man["episodes"][ep] = {"obj": probe["obj"], "frames": rec}
    finally:
        scene.env.close()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(man, open(out, "w"), indent=1)
    return man


def replay(scene, state, probe, masks_root, validity):
    ep, obj = probe["episode"], probe["obj"]
    k_obj = scene.names.index(obj)
    scene.reset()
    apply_state(scene, state)
    p0 = scene.env.object_pos(obj)[:2].copy()
    frames, out = set(probe["frames"]) | {0}, {}
    status = {"expected": 0, "valid": 0, "missing_file": 0, "obj_absent_in_evidence": 0, "robot_occluded": 0,
              "obj_not_rendered": 0}

    def measure(step):
        v = validity["frames"].get(str(step), {"_frame": "missing_file"})
        if step > 0:
            status["expected"] += 1
        if v.get("_frame") == "missing_file":
            if step > 0:
                status["missing_file"] += 1
            out[step] = {"status": "missing_file"}
            return
        mk, seg = np.load(mask_path(masks_root, ep, step)), scene.seg()
        ok = [n for n, s in v.items() if s == "valid"]
        rec = {"iou": {n: iou(seg == k, mk[n]) for k, n in enumerate(scene.names) if n in ok},
               "missing_world_objects": sorted(n for n in ok if n not in scene.names)}
        if step > 0:
            if v.get(obj) == "robot_occluded":
                status["robot_occluded"] += 1
                rec["status"] = "robot_occluded"
            elif v.get(obj) != "valid":
                status["obj_absent_in_evidence"] += 1
                rec["status"] = "obj_absent_in_evidence"
            else:
                c_obs, c_sim = centroid(mk[obj]), centroid(seg == k_obj)
                if c_sim is None:   # valid evidence, candidate lost the object: a failure, not a skip
                    status["obj_not_rendered"] += 1
                    rec["status"] = "obj_not_rendered"
                else:
                    status["valid"] += 1
                    rec["status"] = "valid"
                    rec["obj_centroid_err_px"] = float(np.linalg.norm(c_obs - c_sim))
        out[step] = rec

    if 0 in frames:
        measure(0)
    for k, act in enumerate(probe["actions"], start=1):
        scene.env.step(np.asarray(act))
        if k in frames:
            measure(k)
    disp = (scene.env.object_pos(obj)[:2] - p0).tolist()
    steps = sorted(s for s in out if s > 0)
    static = (float(np.mean(list(out[0]["iou"].values()))) if 0 in out and out[0].get("iou") else None)
    errs = [out[s]["obj_centroid_err_px"] for s in steps if "obj_centroid_err_px" in out[s]]
    dyn = float(np.mean(errs)) if errs else None
    term = [s for s in steps if out[s].get("status") in ("valid", "obj_not_rendered")]
    last = (0.0 if out[term[-1]]["status"] == "obj_not_rendered" else out[term[-1]]["iou"].get(obj)) if term else None
    return {"episode": ep, "static_iou": static, "dyn_err_px": dyn, "dyn_err_px_max": max(errs) if errs else None,
            "final_obj_iou": last, "frame_status": status, "pred_disp": disp,
            "per_frame": {str(s): v for s, v in out.items()}}


def evaluate(world, state_root, probe_root, masks_root, episodes, friction=None, params=None, validity_path=None):
    """friction: legacy {material: mu} (object geoms only); params: {"<mat>.friction|density": v} applied
    with cowm.s6.params.apply_params (includes the table material). validity_path defaults to
    default_validity(masks_root) and must already exist (built once with --build-validity)."""
    from cowm.s6.params import apply_params

    validity_path = validity_path or default_validity(masks_root)
    if not os.path.isfile(validity_path):
        raise FileNotFoundError(f"frozen evidence manifest {validity_path} missing; build it once with "
                                f"g2_replay --build-validity --world <reference world>")
    man = json.load(open(validity_path))
    scene = Scene(world)
    try:
        if params:
            apply_params(scene, params)
        if friction:
            for mat, v in friction.items():
                set_friction(scene, mat, v)
        res = []
        for ep in episodes:
            if ep not in man["episodes"]:
                raise ValueError(f"{ep} has no frozen evidence validity; run --build-validity first")
            state = json.load(open(os.path.join(state_root, ep, "state.json")))["objects"]
            probe = json.load(open(os.path.join(probe_root, ep, "probe.json")))
            res.append(replay(scene, state, probe, masks_root, man["episodes"][ep]))
    finally:
        scene.env.close()
    return res


def summarize(res):
    def mean(key):
        v = [r[key] for r in res if r[key] is not None]
        return float(np.mean(v)) if v else None

    st = {k: int(sum(r["frame_status"][k] for r in res)) for k in res[0]["frame_status"]} if res else {}
    measured = st.get("valid", 0) + st.get("obj_not_rendered", 0)
    usable = st.get("expected", 0) - st.get("missing_file", 0) - st.get("obj_absent_in_evidence", 0) \
        - st.get("robot_occluded", 0)
    s = {"n": len(res), "static_iou": mean("static_iou"), "dyn_err_px": mean("dyn_err_px"),
         "dyn_err_px_p90_episode": (float(np.quantile([r["dyn_err_px"] for r in res if r["dyn_err_px"] is not None],
                                                       0.9)) if any(r["dyn_err_px"] is not None for r in res)
                                    else None),
         "final_obj_iou": mean("final_obj_iou"), "frame_status": st,
         "evidence_coverage": (usable / st["expected"]) if st.get("expected") else None,
         "coverage": (st.get("valid", 0) / st["expected"]) if st.get("expected") else None,
         "episodes_without_dyn": [r["episode"] for r in res if r["dyn_err_px"] is None],
         "measured_frames": measured}
    checks = {"static": s["static_iou"] is not None and s["static_iou"] >= G2_STATIC_IOU,
              "dynamic": s["dyn_err_px"] is not None and s["dyn_err_px"] <= G2_DYN_PX,
              "terminal": s["final_obj_iou"] is not None and s["final_obj_iou"] >= G2_FINAL_IOU,
              "coverage": s["coverage"] is not None and s["coverage"] >= G2_MIN_COVERAGE,
              "every_episode_measured": not s["episodes_without_dyn"]}
    s["checks"] = checks
    s["thresholds"] = {"static_iou": G2_STATIC_IOU, "dyn_err_px": G2_DYN_PX, "final_obj_iou": G2_FINAL_IOU,
                       "coverage": G2_MIN_COVERAGE}
    s["G2_target_evidence"] = bool(all(checks.values()))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--validity", default=None, help="frozen evidence manifest (default: <masks>_validity.json)")
    p.add_argument("--build-validity", action="store_true",
                   help="build the manifest with --world as the reference world, then exit")
    p.add_argument("--friction", default="", help="json {material: mu}")
    p.add_argument("--episodes", default="")
    p.add_argument("--params", default="", help='json {"tabletop.friction": 0.8, ...}')
    a = p.parse_args()
    eps = a.episodes.split(",") if a.episodes else sorted(os.listdir(a.probes))
    validity = a.validity or default_validity(a.masks)
    if a.build_validity:
        if os.path.exists(validity):
            raise SystemExit(f"{validity} exists: the evidence manifest is frozen, not rebuilt")
        man = build_validity(a.world, a.state, a.probes, a.masks, eps, validity)
        print("G2_VALIDITY", len(man["episodes"]), "episodes ->", validity)
        return
    res = evaluate(a.world, a.state, a.probes, a.masks, eps, json.loads(a.friction) if a.friction else None,
                   json.loads(a.params) if a.params else None, validity)
    s = summarize(res)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": s, "episodes": res}, open(a.out, "w"), indent=1)
    print("G2", json.dumps(s))


if __name__ == "__main__":
    main()
