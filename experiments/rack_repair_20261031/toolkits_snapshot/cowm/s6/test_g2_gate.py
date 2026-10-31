"""G2 v2 acceptance-logic tests (pure, no simulator). Every constructed failure must be rejected.
Exit code 1 on any failure.

  good            : 2 complete episodes, small errors          -> pass
  lost_5pct       : candidate loses the object in 5% of valid frames -> fail (lost_frames)
  static_missing  : one episode's step-0 evidence file missing -> fail (every_episode_complete)
  terminal_missing: planned last frame missing, earlier frames fine -> terminal undeterminable, no fallback
  missing_object  : evidence-valid object absent from the world -> IoU 0 + no_missing_world_objects fails
  tail            : 1 of 2 episodes with large dynamic error  -> fail (dynamic_tail)
  tamper          : mask content changed after freezing        -> EvidenceMismatch
  actions_changed : probe actions changed after freezing       -> EvidenceMismatch
  old_schema      : schema-1 manifest                          -> EvidenceMismatch
"""
import json
import os
import sys
import tempfile

import numpy as np

from cowm.s6 import g2_replay as g

NAMES = ["bowl", "plate"]
STEPS = list(range(0, 21))   # frame 0 + 20 probe frames


def validity(missing_steps=(), obj_absent=()):
    fr = {}
    for s in STEPS:
        if s in missing_steps:
            fr[str(s)] = {"_frame": "missing_file"}
        else:
            fr[str(s)] = {"bowl": "empty" if s in obj_absent else "valid", "plate": "valid"}
    return {"obj": "bowl", "frames_planned": STEPS, "terminal_step": STEPS[-1], "frames": fr}


def episode(name, v, err=2.0, lost=(), world=NAMES, iou=0.8):
    pf = {}
    for s in STEPS:
        lost_s = s in lost
        pf[s] = g.frame_record(v, s, "bowl", world, {n: iou for n in world}, not lost_s,
                               None if lost_s else err)
    return g.episode_metrics(name, "bowl", v["terminal_step"], pf)


def summary(*eps):
    return g.summarize(list(eps))


def main():
    res = {}
    v = validity()
    s = summary(episode("e0", v), episode("e1", v))
    res["good"] = s["G2_target_evidence"]
    # 1 lost frame of 20 per episode = 5% > G2_MAX_LOST (2%)
    s = summary(episode("e0", v, lost=(7,)), episode("e1", v, lost=(9,)))
    res["lost_5pct"] = (not s["G2_target_evidence"]) and not s["checks"]["lost_frames"] and \
        s["dyn_err_px"] > g.G2_DYN_PX
    s = summary(episode("e0", validity(missing_steps=(0,))), episode("e1", v))
    res["static_missing"] = (not s["G2_target_evidence"]) and s["incomplete_episodes"] == ["e0"]
    e = episode("e0", validity(missing_steps=(20,)))
    s = summary(e, episode("e1", v))
    res["terminal_missing"] = (e["final_obj_iou"] is None and e["terminal_status"] == "missing_file"
                               and not s["G2_target_evidence"] and s["terminal_undeterminable"] == ["e0"])
    e = episode("e0", v, world=["bowl"])   # plate valid in evidence, absent from the world
    s = summary(e, episode("e1", v))
    res["missing_object"] = (abs(e["static_iou"] - 0.4) < 1e-9 and e["missing_world_objects"] == ["plate"]
                             and not s["checks"]["no_missing_world_objects"] and not s["G2_target_evidence"])
    s = summary(episode("e0", v), episode("e1", v, err=20.0))
    res["tail"] = (not s["G2_target_evidence"]) and not s["checks"]["dynamic_tail"]
    # manifest binding on real files
    with tempfile.TemporaryDirectory() as d:
        probes, masks = os.path.join(d, "probes"), os.path.join(d, "masks")
        ep = "ep0"
        os.makedirs(os.path.join(probes, ep))
        probe = {"episode": ep, "obj": "bowl", "actions": [[0.0] * 7] * 3, "frames": [1, 2, 3]}
        json.dump(probe, open(os.path.join(probes, ep, "probe.json"), "w"))
        shas = {}
        for s_ in [0, 1, 2, 3]:
            p = g.mask_path(masks, ep, s_)
            os.makedirs(os.path.dirname(p))
            np.savez(p, bowl=np.ones((4, 4), bool))
            shas[str(s_)] = g._sha_file(p)
        man = {"schema": g.SCHEMA, "robot_frac": g.ROBOT_FRAC, "episodes": {ep: {
            "obj": "bowl", "frames_planned": [0, 1, 2, 3], "terminal_step": 3,
            "actions_sha256": g._sha_json(probe["actions"]), "mask_sha256": shas,
            "frames": {str(s_): {"bowl": "valid"} for s_ in [0, 1, 2, 3]}}}}

        def rejects(m):
            try:
                g.verify_manifest(m, probes, masks, [ep])
                return False
            except g.EvidenceMismatch:
                return True

        res["clean_manifest_ok"] = not rejects(man)
        np.savez(g.mask_path(masks, ep, 2), bowl=np.zeros((4, 4), bool))
        res["tamper"] = rejects(man)
        np.savez(g.mask_path(masks, ep, 2), bowl=np.ones((4, 4), bool))
        man["episodes"][ep]["mask_sha256"]["2"] = g._sha_file(g.mask_path(masks, ep, 2))
        probe["actions"][1] = [0.1] * 7
        json.dump(probe, open(os.path.join(probes, ep, "probe.json"), "w"))
        res["actions_changed"] = rejects(man)
        res["old_schema"] = rejects(dict(man, schema=1))
        probe["actions"][1] = [0.0] * 7   # restore actions, then change only the probe object
        probe["obj"] = "plate"
        json.dump(probe, open(os.path.join(probes, ep, "probe.json"), "w"))
        res["obj_changed"] = rejects(man)
        probe["obj"] = "bowl"
        json.dump(probe, open(os.path.join(probes, ep, "probe.json"), "w"))
        res["clean_again_ok"] = not rejects(man)
        res["rule_changed"] = rejects(dict(man, robot_frac=0.7))
        wdir = os.path.join(d, "refworld")
        os.makedirs(wdir)
        json.dump({"objects": []}, open(os.path.join(wdir, "world.json"), "w"))
        man_ref = dict(man, ref_world=wdir, ref_world_json_sha256=g._sha_file(os.path.join(wdir, "world.json")))
        res["ref_world_ok"] = not rejects(man_ref)
        json.dump({"objects": [1]}, open(os.path.join(wdir, "world.json"), "w"))
        res["ref_world_changed"] = rejects(man_ref)
    ok = all(res.values())
    print(" ".join(f"{k}={v}" for k, v in res.items()))
    print("G2_GATE_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
