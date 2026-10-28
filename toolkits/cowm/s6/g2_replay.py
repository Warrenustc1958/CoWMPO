"""S6 G2 target-evidence gate + replay metrics (Builder side, plan §9.7-D).

Replays the probe actions the Target executed in the code world (S5 state applied) and compares the
world's segmentation render with SAM2 masks of the returned probe frames:
  static  = mean IoU over observed objects at step 0
  dynamic = probe-object mask-centroid error (px) over the probe frames, and its final-frame IoU
G2 passes iff static >= G2_STATIC_IOU and dynamic_err_px <= G2_DYN_PX (thresholds fixed before fitting).
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


def set_friction(scene, material, value):
    m = scene.env.sim.model
    for k, o in enumerate(scene.world["objects"]):
        if o.get("material") == material:
            m.geom_friction[scene.geom2obj == k, 0] = value


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def replay(scene, state, probe, masks_root):
    ep, obj = probe["episode"], probe["obj"]
    k_obj = scene.names.index(obj)
    scene.reset()
    apply_state(scene, state)
    p0 = scene.env.object_pos(obj)[:2].copy()
    frames, out = set(probe["frames"]), {}

    def measure(step):
        f = os.path.join(masks_root, f"{ep}__f{step:03d}", "masks.npz")
        if not os.path.isfile(f):
            return
        mk = np.load(f)
        seg, rob = scene.seg(), scene.robot_mask()
        # same rule as S5: a mask that is mostly robot (rendered from the shared prior) is a perception error
        ok = [n for n in mk.files if mk[n].sum() and (mk[n] & rob).sum() / mk[n].sum() <= 0.5]
        rec = {"iou": {n: iou(seg == k, mk[n]) for k, n in enumerate(scene.names) if n in ok}}
        if obj in ok:
            c_obs, c_sim = centroid(mk[obj]), centroid(seg == k_obj)
            rec["obj_centroid_err_px"] = (float(np.linalg.norm(c_obs - c_sim))
                                          if c_obs is not None and c_sim is not None else None)
        out[step] = rec

    if 0 in frames:
        measure(0)
    for k, act in enumerate(probe["actions"], start=1):
        scene.env.step(np.asarray(act))
        if k in frames:
            measure(k)
    disp = (scene.env.object_pos(obj)[:2] - p0).tolist()
    steps = sorted(out)
    static = float(np.mean(list(out[0]["iou"].values()))) if 0 in out and out[0]["iou"] else None
    errs = [out[s].get("obj_centroid_err_px") for s in steps[1:]]
    errs = [e for e in errs if e is not None]
    dyn = float(np.mean(errs)) if errs else None
    last = out[steps[-1]]["iou"].get(obj) if steps else None
    return {"episode": ep, "static_iou": static, "dyn_err_px": dyn, "final_obj_iou": last,
            "pred_disp": disp, "per_frame": {str(s): v for s, v in out.items()}}


def evaluate(world, state_root, probe_root, masks_root, episodes, friction=None, params=None):
    """friction: legacy {material: mu} (object geoms only); params: {"<mat>.friction|density": v} applied
    with cowm.s6.params.apply_params (includes the table material)."""
    from cowm.s6.params import apply_params

    scene = Scene(world)
    if params:
        apply_params(scene, params)
    if friction:
        for mat, v in friction.items():
            set_friction(scene, mat, v)
    res = []
    for ep in episodes:
        state = json.load(open(os.path.join(state_root, ep, "state.json")))["objects"]
        probe = json.load(open(os.path.join(probe_root, ep, "probe.json")))
        res.append(replay(scene, state, probe, masks_root))
    return res


def summarize(res):
    f = lambda key: float(np.mean([r[key] for r in res if r[key] is not None])) if any(
        r[key] is not None for r in res) else None
    s = {"n": len(res), "static_iou": f("static_iou"), "dyn_err_px": f("dyn_err_px"),
         "final_obj_iou": f("final_obj_iou")}
    s["G2_target_evidence"] = bool(s["static_iou"] is not None and s["dyn_err_px"] is not None and
                                   s["static_iou"] >= G2_STATIC_IOU and s["dyn_err_px"] <= G2_DYN_PX)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--friction", default="", help="json {material: mu}")
    p.add_argument("--episodes", default="")
    p.add_argument("--params", default="", help='json {"tabletop.friction": 0.8, ...}')
    a = p.parse_args()
    eps = a.episodes.split(",") if a.episodes else sorted(os.listdir(a.probes))
    res = evaluate(a.world, a.state, a.probes, a.masks, eps, json.loads(a.friction) if a.friction else None,
                   json.loads(a.params) if a.params else None)
    s = summarize(res)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": s, "episodes": res}, open(a.out, "w"), indent=1)
    print("G2", s)


if __name__ == "__main__":
    main()
