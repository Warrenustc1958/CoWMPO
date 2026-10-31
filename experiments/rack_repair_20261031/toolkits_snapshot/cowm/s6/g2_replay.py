"""S6 G2 target-evidence gate + replay metrics (Builder side, plan §9.7-D). Manifest schema 2 (audit 10.29).

Replays the probe actions the Target executed in the code world (S5 state applied) and compares the
world's segmentation render with SAM2 masks of the returned probe frames.

Evidence (decided ONCE, frozen, content-bound):
  build_validity() runs one designated reference world and records, per expected frame and object,
  valid | missing_file | empty | robot_occluded, plus the sha256 of every mask file, of the probe actions and
  frame list, of the reference world.json, and the code version. evaluate() verifies all hashes before
  scoring: re-run perception or changed probes need a NEW manifest (EvidenceMismatch otherwise).

Per episode (episode_metrics, pure):
  static   = mean IoU at step 0 over every evidence-valid object; an object valid in evidence but absent from
             the world counts as IoU 0 and is listed in missing_world_objects (a structural failure)
  dynamic  = probe-object centroid error over probe frames; a frame whose evidence is valid but where the
             candidate does not render the object is a prediction failure: counted in lost_frames and scored
             as LOST_PX in the error mean (never silently dropped)
  terminal = probe-object IoU at the PLANNED last probe frame only; if that frame's evidence is unusable the
             terminal is "undeterminable" (no fallback to an earlier frame)
  complete = static, dynamic and terminal all determinable

G2 passes iff every check holds (thresholds fixed before the gate split is used):
  static_iou >= G2_STATIC_IOU, dyn_err_px <= G2_DYN_PX, frac episodes with dyn > G2_DYN_PX <= G2_DYN_TAIL,
  final_obj_iou >= G2_FINAL_IOU, lost-frame rate <= G2_MAX_LOST, evidence coverage >= G2_MIN_COVERAGE,
  every episode complete, no missing world objects.
"""
import argparse
import hashlib
import json
import os

import numpy as np

SCHEMA = 2
G2_STATIC_IOU = 0.6
G2_DYN_PX = 6.0
G2_DYN_TAIL = 0.2         # at most this fraction of episodes may exceed G2_DYN_PX (10.29)
G2_FINAL_IOU = 0.5
G2_MAX_LOST = 0.02        # valid-evidence frames where the candidate lost the probe object (10.29)
G2_MIN_COVERAGE = 0.9     # usable evidence frames / expected probe frames (property of the evidence)
ROBOT_FRAC = 0.5          # evidence mask mostly covered by the robot -> perception error (same rule as S5)
LOST_PX = 362.0           # error assigned to a lost frame = diagonal of the 256x256 policy image


class EvidenceMismatch(RuntimeError):
    pass


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sha_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def _code_sha():
    return _sha_file(os.path.abspath(__file__))


def set_friction(scene, material, value):
    """Incremental per-material sliding-friction override (used by the L1 backend and old scripts).
    material: a material handle (e.g. "ceramic", "tabletop") or the class "table_surface" for the table.
    Object materials also update their object-table <pair>s (contact rule object_table_pair), so the value
    reaches the contact that actually controls sliding. Raises on unknown materials and invalid values."""
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Sliding friction must be finite and non-negative")
    m = scene.env.sim.model
    selected = np.zeros(m.ngeom, dtype=bool)
    if material in ("table_surface", scene.world["table"].get("material")):
        for g in range(m.ngeom):
            if (m.geom_id2name(g) or "").startswith("table_collision"):
                selected[g] = True
    objs = [k for k, o in enumerate(scene.world["objects"]) if o.get("material") == material]
    for k in objs:
        selected |= scene.geom2obj == k
    if not selected.any():
        raise ValueError(f"No collision geometry uses material {material!r}")
    m.geom_friction[selected, 0] = value
    for p in range(m.npair):
        if any(int(scene.geom2obj[g]) in objs for g in (int(m.pair_geom1[p]), int(m.pair_geom2[p]))):
            m.pair_friction[p, 0:2] = value


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def mask_path(masks_root, ep, step):
    return os.path.join(masks_root, f"{ep}__f{step:03d}", "masks.npz")


def default_validity(masks_root):
    """Frozen manifest lives next to the probe masks it describes (one per evidence set)."""
    return os.path.normpath(masks_root) + f"_validity_v{SCHEMA}.json"


def probe_frames(probe):
    return sorted(set(probe["frames"]) | {0})


def build_validity(ref_world, state_root, probe_root, masks_root, episodes, out):
    from cowm.init.estimate_state import Scene
    from cowm.s6.plan_probe import apply_state
    from cowm.world.artifact import tree_sha, verify_tree
    from cowm.init.observation import load_association, select_masks

    ref_hash = tree_sha(ref_world)
    scene = Scene(ref_world)
    man = {"schema": SCHEMA, "ref_world": ref_world,
           "ref_world_tree_sha256": ref_hash, "ref_state_root": os.path.abspath(state_root),
           "ref_world_json_sha256": _sha_file(os.path.join(ref_world, "world.json")),
           "masks_root": os.path.abspath(masks_root), "robot_frac": ROBOT_FRAC, "code_sha256": _code_sha(),
           "episodes": {}}
    try:
        for ep in episodes:
            state_path = os.path.join(state_root, ep, "state.json")
            state_hash = _sha_file(state_path)
            state_doc = json.load(open(state_path))
            if (state_doc.get("episode") != ep or state_doc.get("world_tree_sha256") != ref_hash or
                    state_doc.get("world_sha256") != scene.world["world_sha256"]):
                raise EvidenceMismatch(f"{ep}: reference state does not belong to the frozen reference world")
            state = state_doc["objects"]
            probe = json.load(open(os.path.join(probe_root, ep, "probe.json")))
            frames = probe_frames(probe)
            scene.reset()
            if state_doc.get("robot_required") and state_doc.get("robot") is None:
                raise EvidenceMismatch(f"{ep}: required robot initialization is missing")
            apply_state(scene, state, require_matched=True, robot=state_doc.get("robot"))
            rec, shas, association_shas = {}, {}, {}

            def check(step):
                f = mask_path(masks_root, ep, step)
                if not os.path.isfile(f):
                    rec[str(step)] = {"_frame": "missing_file"}
                    return
                shas[str(step)] = _sha_file(f)
                frame_ep = f"{ep}__f{step:03d}"
                association = load_association(os.path.dirname(f), frame_ep)
                with np.load(f, allow_pickle=False) as archive:
                    mk = {n: archive[n] for n in archive.files}
                _, identities = select_masks(scene.names, mk, association)
                association_shas[str(step)] = _sha_file(os.path.join(os.path.dirname(f), "association.json"))
                rob = scene.robot_mask()
                st = {}
                for n, identity in identities.items():
                    if identity != "matched":
                        st[n] = "identity_" + identity
                        continue
                    s = int(mk[n].sum())
                    st[n] = "empty" if s == 0 else ("robot_occluded" if (mk[n] & rob).sum() / s > ROBOT_FRAC
                                                    else "valid")
                rec[str(step)] = st

            check(0)
            for k, act in enumerate(probe["actions"], start=1):
                scene.env.step(np.asarray(act))
                if k in frames:
                    check(k)
            man["episodes"][ep] = {"obj": probe["obj"], "frames_planned": frames, "terminal_step": frames[-1],
                                   "ref_state_sha256": state_hash,
                                   "actions_sha256": _sha_json(probe["actions"]), "mask_sha256": shas,
                                   "association_sha256": association_shas,
                                   "frames": rec}
    finally:
        scene.env.close()
    verify_tree(ref_world, ref_hash)
    for ep, e in man["episodes"].items():
        if _sha_file(os.path.join(state_root, ep, "state.json")) != e["ref_state_sha256"]:
            raise EvidenceMismatch(f"{ep}: reference state changed during manifest construction")
    verify_manifest(man, probe_root, masks_root, episodes)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(man, open(out, "w"), indent=1)
    return man


def verify_manifest(man, probe_root, masks_root, episodes):
    """Raise EvidenceMismatch unless manifest, probes and mask files are exactly the frozen evidence."""
    if man.get("schema") != SCHEMA:
        raise EvidenceMismatch(f"manifest schema {man.get('schema')} != {SCHEMA}: build a new manifest")
    if man.get("robot_frac") != ROBOT_FRAC:
        raise EvidenceMismatch(f"validity rule changed (robot_frac {man.get('robot_frac')} != {ROBOT_FRAC})")
    ref = man.get("ref_world")
    if ref is not None:   # the reference world that decided validity must be unchanged
        wj = os.path.join(ref, "world.json")
        if not os.path.isfile(wj) or _sha_file(wj) != man.get("ref_world_json_sha256"):
            raise EvidenceMismatch(f"reference world {ref} missing or changed since the manifest was frozen")
        if man.get("ref_world_tree_sha256"):
            from cowm.world.artifact import verify_tree
            try:
                verify_tree(ref, man["ref_world_tree_sha256"])
            except ValueError as exc:
                raise EvidenceMismatch(str(exc)) from exc
    for ep in episodes:
        e = man["episodes"].get(ep)
        if e is None:
            raise EvidenceMismatch(f"{ep}: not in the frozen manifest")
        if man.get("ref_state_root"):
            state_path = os.path.join(man["ref_state_root"], ep, "state.json")
            if not os.path.isfile(state_path) or _sha_file(state_path) != e.get("ref_state_sha256"):
                raise EvidenceMismatch(f"{ep}: reference state missing or changed since freezing")
        probe = json.load(open(os.path.join(probe_root, ep, "probe.json")))
        if probe.get("obj") != e["obj"]:
            raise EvidenceMismatch(f"{ep}: probe object {probe.get('obj')!r} != frozen {e['obj']!r}")
        if _sha_json(probe["actions"]) != e["actions_sha256"] or probe_frames(probe) != e["frames_planned"]:
            raise EvidenceMismatch(f"{ep}: probe actions/frames differ from the frozen evidence")
        for step, st in e["frames"].items():
            f = mask_path(masks_root, ep, int(step))
            if "_frame" in st:
                if os.path.isfile(f):
                    raise EvidenceMismatch(f"{ep} f{step}: mask appeared after freezing")
            elif not os.path.isfile(f) or _sha_file(f) != e["mask_sha256"][step]:
                raise EvidenceMismatch(f"{ep} f{step}: mask missing or content changed")
            if "association_sha256" in e and "_frame" not in st:
                sidecar = os.path.join(os.path.dirname(f), "association.json")
                if not os.path.isfile(sidecar) or _sha_file(sidecar) != e["association_sha256"].get(step):
                    raise EvidenceMismatch(f"{ep} f{step}: association missing or changed after freezing")


def require_reference_bindings(man, episodes):
    """S6 fitting requires assets, reference states and identity decisions bound before candidate evaluation."""
    if not man.get("ref_world_tree_sha256") or not man.get("ref_state_root") or not man.get("ref_world"):
        raise EvidenceMismatch("S6 requires reference asset/state bindings; freeze a new manifest file")
    for ep in episodes:
        e = man.get("episodes", {}).get(ep, {})
        if not e.get("ref_state_sha256") or "association_sha256" not in e:
            raise EvidenceMismatch(f"{ep}: S6 requires frozen reference-state and frame-identity bindings")


def frame_record(validity, step, obj, world_names, seg_iou, obj_rendered, centroid_err):
    """One frame's outcome given the frozen validity. seg_iou: {name: IoU} for every world object;
    obj_rendered/centroid_err: probe object in the candidate's render. Pure (used by replay and tests)."""
    v = validity["frames"].get(str(step), {"_frame": "missing_file"})
    if v.get("_frame") == "missing_file":
        return {"status": "missing_file"}
    valid = [n for n, s in v.items() if s == "valid"]
    rec = {"iou": {n: (seg_iou.get(n, 0.0) if n in world_names else 0.0) for n in valid},
           "uncertain_identities": sorted(n for n, status in v.items() if status.startswith("identity_")),
           "missing_world_objects": sorted(n for n in valid if n not in world_names)}
    if step == 0:
        rec["status"] = "static"
    elif v.get(obj) in ("identity_ambiguous", "identity_missing"):
        rec["status"] = v[obj]
    elif v.get(obj) == "robot_occluded":
        rec["status"] = "robot_occluded"
    elif v.get(obj) != "valid":
        rec["status"] = "obj_absent_in_evidence"
    elif not obj_rendered:
        rec["status"] = "obj_not_rendered"
    else:
        rec["status"] = "valid"
        rec["obj_centroid_err_px"] = float(centroid_err)
    return rec


def episode_metrics(ep, obj, terminal_step, per_frame):
    """per_frame: {step: frame_record}. Pure: everything a gate decision needs for one episode."""
    st = {"expected": 0, "valid": 0, "missing_file": 0, "obj_absent_in_evidence": 0, "robot_occluded": 0,
          "identity_ambiguous": 0, "identity_missing": 0,
          "obj_not_rendered": 0}
    errs = []
    for s in sorted(k for k in per_frame if k > 0):
        r = per_frame[s]
        st["expected"] += 1
        st[r["status"]] += 1
        if r["status"] == "valid":
            errs.append(r["obj_centroid_err_px"])
        elif r["status"] == "obj_not_rendered":
            errs.append(LOST_PX)
    f0 = per_frame.get(0, {})
    static = float(np.mean(list(f0["iou"].values()))) if f0.get("iou") else None
    t = per_frame.get(terminal_step, {})
    if t.get("status") == "valid":
        term = t["iou"].get(obj)
    elif t.get("status") == "obj_not_rendered":
        term = 0.0
    else:
        term = None   # undeterminable: no fallback to an earlier frame
    missing = sorted({n for r in per_frame.values() for n in r.get("missing_world_objects", [])})
    dyn = float(np.mean(errs)) if errs else None
    return {"episode": ep, "obj": obj, "static_iou": static, "dyn_err_px": dyn,
            "dyn_err_px_max": max(errs) if errs else None, "final_obj_iou": term,
            "terminal_step": terminal_step, "terminal_status": t.get("status", "missing_file"),
            "missing_world_objects": missing, "frame_status": st,
            "initial_state_valid": True, "initial_state_issues": {},
            "static_uncertain_identities": f0.get("uncertain_identities", []),
            "complete": static is not None and dyn is not None and term is not None and
                        not f0.get("uncertain_identities")}


def invalid_initial_state(ep, validity, world_names, issues):
    """Keep every planned frame and evidence denominator when initialization is untrustworthy."""
    frames = {s: frame_record(validity, s, validity["obj"], world_names, {}, False, None)
              for s in validity["frames_planned"]}
    result = episode_metrics(ep, validity["obj"], validity["terminal_step"], frames)
    result.update(initial_state_valid=False, initial_state_issues=issues, complete=False,
                  per_frame={str(s): f for s, f in frames.items()})
    return result


def replay(scene, state, probe, masks_root, validity, robot=None):
    from cowm.init.estimate_state import iou
    from cowm.s6.plan_probe import apply_state

    ep, obj = probe["episode"], probe["obj"]
    k_obj = scene.names.index(obj)
    scene.reset()
    apply_state(scene, state, require_matched=True, robot=robot)
    p0 = scene.env.object_pos(obj)[:2].copy()
    frames, out = set(probe_frames(probe)), {}

    def measure(step):
        f = mask_path(masks_root, ep, step)
        if not os.path.isfile(f):
            out[step] = frame_record(validity, step, obj, scene.names, {}, False, None)
            return
        mk, seg = np.load(f), scene.seg()
        seg_iou = {n: iou(seg == k, mk[n]) for k, n in enumerate(scene.names) if n in mk.files}
        c_obs, c_sim = centroid(mk[obj]) if obj in mk.files else None, centroid(seg == k_obj)
        err = float(np.linalg.norm(c_obs - c_sim)) if c_obs is not None and c_sim is not None else None
        out[step] = frame_record(validity, step, obj, scene.names, seg_iou, c_sim is not None, err)

    measure(0)
    for k, act in enumerate(probe["actions"], start=1):
        scene.env.step(np.asarray(act))
        if k in frames:
            measure(k)
    m = episode_metrics(ep, obj, validity["terminal_step"], out)
    m["pred_disp"] = (scene.env.object_pos(obj)[:2] - p0).tolist()
    m["per_frame"] = {str(s): v for s, v in out.items()}
    return m


def evaluate(world, state_root, probe_root, masks_root, episodes, friction=None, params=None, validity_path=None,
             world_tree_sha256=None, require_bound_reference=False):
    """friction: legacy {material: mu}, applied through apply_params as {"<mat>.friction": mu};
    params: {"<mat>.friction|density": v}. validity_path defaults to default_validity(masks_root)."""
    from cowm.init.estimate_state import Scene
    from cowm.s6.params import apply_params
    from cowm.init.observation import state_issues
    from cowm.init.robot_state import robot_issues
    from cowm.world.artifact import verify_tree

    if world_tree_sha256 is not None:
        verify_tree(world, world_tree_sha256)

    validity_path = validity_path or default_validity(masks_root)
    if not os.path.isfile(validity_path):
        raise FileNotFoundError(f"frozen evidence manifest {validity_path} missing; build it once with "
                                f"g2_replay --build-validity --world <reference world>")
    man = json.load(open(validity_path))
    if require_bound_reference:
        require_reference_bindings(man, episodes)
    verify_manifest(man, probe_root, masks_root, episodes)
    values = dict(params or {})
    values.update({f"{mat}.friction": float(v) for mat, v in (friction or {}).items()})
    scene = Scene(world)
    try:
        if values:
            apply_params(scene, values)
        res = []
        for ep in episodes:
            document = json.load(open(os.path.join(state_root, ep, "state.json")))
            state = document["objects"]
            probe = json.load(open(os.path.join(probe_root, ep, "probe.json")))
            issues = state_issues(state, scene.names)
            invalid_robot = robot_issues(document.get("robot"))
            if document.get("robot_required") and document.get("robot") is None:
                invalid_robot.append("required_robot_state_missing")
            if invalid_robot:
                issues["_robot"] = invalid_robot
            if document.get("episode") != ep:
                issues["_episode"] = ["state_episode_mismatch"]
            if document.get("world_sha256") != scene.world["world_sha256"]:
                issues["_world"] = ["state_world_mismatch"]
            if world_tree_sha256 is not None and document.get("world_tree_sha256") != world_tree_sha256:
                issues["_world_tree"] = ["state_world_assets_mismatch"]
            if probe["obj"] not in scene.names:
                issues[probe["obj"]] = ["probe_object_absent_from_world"]
            if issues:
                res.append(invalid_initial_state(ep, man["episodes"][ep], scene.names, issues))
                continue
            result = replay(scene, state, probe, masks_root, man["episodes"][ep],
                            robot=document.get("robot"))
            result["robot_initialization"] = "estimated" if document.get("robot") else "fixed_reset_prior"
            res.append(result)
        res_info = {"applied_params": values, "contact_rule": scene.world["table"].get("contact_rule", "legacy"),
                    "validity": os.path.abspath(validity_path), "world_tree_sha256": world_tree_sha256}
    finally:
        scene.env.close()
    if world_tree_sha256 is not None:
        verify_tree(world, world_tree_sha256)
    for r in res:
        r["run"] = res_info
    return res


def summarize(res):
    def mean(key):
        v = [r[key] for r in res if r[key] is not None]
        return float(np.mean(v)) if v else None

    keys = {k for r in res for k in r["frame_status"]}
    st = {k: int(sum(r["frame_status"].get(k, 0) for r in res)) for k in sorted(keys)}
    exp = st.get("expected", 0)
    unusable = sum(st.get(k, 0) for k in ("missing_file", "obj_absent_in_evidence", "robot_occluded",
                                        "identity_ambiguous", "identity_missing"))
    usable = exp - unusable
    dyn = [r["dyn_err_px"] for r in res if r["dyn_err_px"] is not None]
    s = {"n": len(res), "static_iou": mean("static_iou"), "dyn_err_px": mean("dyn_err_px"),
         "dyn_err_px_p90_episode": float(np.quantile(dyn, 0.9)) if dyn else None,
         "dyn_tail_frac": float(np.mean([d > G2_DYN_PX for d in dyn])) if dyn else None,
         "final_obj_iou": mean("final_obj_iou"), "frame_status": st,
         "evidence_coverage": usable / exp if exp else None,
         "lost_rate": st.get("obj_not_rendered", 0) / usable if usable else None,
         "incomplete_episodes": [r["episode"] for r in res if not r["complete"]],
         "invalid_initial_states": {r["episode"]: r.get("initial_state_issues", {}) for r in res
                                    if not r.get("initial_state_valid", True)},
         "missing_world_objects": sorted({n for r in res for n in r["missing_world_objects"]}),
         "terminal_undeterminable": [r["episode"] for r in res if r["final_obj_iou"] is None]}
    checks = {"static": s["static_iou"] is not None and s["static_iou"] >= G2_STATIC_IOU,
              "dynamic": s["dyn_err_px"] is not None and s["dyn_err_px"] <= G2_DYN_PX,
              "dynamic_tail": s["dyn_tail_frac"] is not None and s["dyn_tail_frac"] <= G2_DYN_TAIL,
              "terminal": s["final_obj_iou"] is not None and s["final_obj_iou"] >= G2_FINAL_IOU,
              "lost_frames": s["lost_rate"] is not None and s["lost_rate"] <= G2_MAX_LOST,
              "evidence_coverage": s["evidence_coverage"] is not None and s["evidence_coverage"] >= G2_MIN_COVERAGE,
              "every_episode_complete": bool(res) and not s["incomplete_episodes"],
              "initial_states_valid": bool(res) and not s["invalid_initial_states"],
              "finite_metrics": bool(res) and all(
                  r.get(k) is not None and np.isfinite(r[k]) and
                  (0 <= r[k] <= 1 if k != "dyn_err_px" else r[k] >= 0)
                  for r in res for k in ("static_iou", "dyn_err_px", "final_obj_iou")),
              "no_missing_world_objects": not s["missing_world_objects"]}
    s["checks"] = checks
    s["thresholds"] = {"static_iou": G2_STATIC_IOU, "dyn_err_px": G2_DYN_PX, "dyn_tail_frac": G2_DYN_TAIL,
                       "final_obj_iou": G2_FINAL_IOU, "lost_rate": G2_MAX_LOST, "coverage": G2_MIN_COVERAGE,
                       "lost_px": LOST_PX}
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
    p.add_argument("--world-tree-sha256", help="frozen compiled directory hash (required by S6 identification)")
    p.add_argument("--require-bound-reference", action="store_true",
                   help="refuse a legacy manifest without reference asset/state hashes")
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
                   json.loads(a.params) if a.params else None, validity, a.world_tree_sha256,
                   a.require_bound_reference)
    s = summarize(res)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": s, "episodes": res}, open(a.out, "w"), indent=1)
    print("G2", json.dumps(s))


if __name__ == "__main__":
    main()
