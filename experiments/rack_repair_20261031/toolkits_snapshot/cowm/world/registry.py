"""CoWM-PO world version registry (plan §9.7): immutable, append-only world versions.

Every compiled world is registered with its parent version, what changed, which evidence it saw, its
verifier report and cost. Status levels (registry schema 2, audit 10.29):
  rejected    : the report is not ok, or G0/G1 failed (a failed structural check is a rejection even if the
                G0/G1 gate dict was not recomputed by the caller)
  executable  : G0 build + G1 self-consistency pass and report ok -> may be loaded for debugging rollouts
  g2_verified : promoted with a passing G2 target-evidence summary (cowm.s6.g2_replay.summarize)
  frozen      : promoted for method experiments (code/SDK/assets/params/state estimator fixed); only frozen
                versions may enter the formal gate / training comparisons
The registry hashes EVERY file of the world directory (world.json, XML, meshes); resolve() refuses a world
whose files changed after registration. "published" (schema 1) is read as "executable".
"""
import fcntl
import hashlib
import json
import os
import time

REGISTRY = os.environ.get("COWM_REGISTRY", "/data_cf/cowmpo/worlds/world_registry.json")
LEVELS = ("rejected", "executable", "g2_verified", "frozen")


def _load(path):
    if not os.path.isfile(path):
        return {"schema": "cowm-registry-2", "versions": []}
    with open(path) as f:
        return json.load(f)


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_sha(world_dir):
    """sha256 over (relative path, file sha256) of every file in the world directory, sorted."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(world_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, world_dir).encode() + b"\0" + _sha_file(p).encode() + b"\n")
    return h.hexdigest()


def _level(v):
    s = v["status"]
    return "executable" if s == "published" else s


def _write(reg, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, path)


def register(world_dir, parent=None, source="", evidence_ids=(), changed=None, report=None, cost=None,
             path=REGISTRY):
    """Append one version. Returns its version id (e.g. 'w0007')."""
    world_json = os.path.join(world_dir, "world.json")
    world = json.load(open(world_json)) if os.path.isfile(world_json) else {}
    report = report or {}
    gates = report.get("gates", {})
    executable = bool(report.get("ok") and gates.get("G0_build") and gates.get("G1_self_consistent"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        reg = _load(path)
        if parent is not None and parent not in {v["version"] for v in reg["versions"]}:
            raise ValueError(f"unknown parent version {parent!r}")
        vid = f"w{len(reg['versions']):04d}"
        reg["versions"].append({
            "version": vid, "parent": parent, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "world_dir": os.path.abspath(world_dir), "source": source,
            "program_sha256": world.get("program_sha256"), "world_sha256": world.get("world_sha256"),
            "world_json_sha256": _sha_file(world_json) if os.path.isfile(world_json) else None,
            "tree_sha256": tree_sha(world_dir) if os.path.isdir(world_dir) else None,
            "sdk_version": world.get("sdk_version"),
            "contact_rule": (world.get("table") or {}).get("contact_rule", "legacy"),
            "evidence_ids": list(evidence_ids), "changed": changed or {}, "gates": gates,
            "report_ok": bool(report.get("ok")), "stage": report.get("stage"),
            "errors": report.get("errors", []), "cost": cost or {},
            "status": "executable" if executable else "rejected", "promotions": []})
        reg["schema"] = "cowm-registry-2"
        _write(reg, path)
    return vid


def promote(version, level, evidence, path=REGISTRY):
    """Raise a version to g2_verified (evidence = passing G2 summary) or frozen (evidence = freeze record
    with at least {"frozen_by", "config"}). Levels only go up, one step at a time; files must be unchanged."""
    if level not in ("g2_verified", "frozen"):
        raise ValueError(f"cannot promote to {level!r}")
    if level == "g2_verified" and not (evidence or {}).get("G2_target_evidence"):
        raise ValueError("g2_verified requires a G2 summary with G2_target_evidence=True")
    if level == "frozen" and not {"frozen_by", "config"} <= set(evidence or {}):
        raise ValueError("frozen requires a freeze record with 'frozen_by' and 'config'")
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        reg = _load(path)
        v = next((x for x in reg["versions"] if x["version"] == version), None)
        if v is None:
            raise KeyError(f"world version {version!r} not in registry")
        cur = _level(v)
        if LEVELS.index(level) != LEVELS.index(cur) + 1 or cur == "rejected":
            raise ValueError(f"{version}: cannot go from {cur} to {level}")
        _check_files(v)
        v["status"] = level
        v.setdefault("promotions", []).append({"level": level, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                               "evidence": evidence})
        _write(reg, path)


def _check_files(v):
    wj = os.path.join(v["world_dir"], "world.json")
    if not os.path.isfile(wj) or _sha_file(wj) != v["world_json_sha256"]:
        raise ValueError(f"world {v['version']} was modified after registration (world.json hash mismatch)")
    if v.get("tree_sha256") is None:
        raise ValueError(f"world {v['version']} has no file-tree hash (schema-1 entry): re-register it")
    if tree_sha(v["world_dir"]) != v["tree_sha256"]:
        raise ValueError(f"world {v['version']} files (XML/mesh) changed after registration")


def resolve(version, path=REGISTRY, min_level="executable", require_published=None):
    """Return the world_dir of a version after checking its level and that no file changed.
    require_published (schema-1 callers): True -> min_level executable, False -> any level."""
    if require_published is not None:
        min_level = "executable" if require_published else "rejected"
    reg = _load(path)
    v = next((x for x in reg["versions"] if x["version"] == version), None)
    if v is None:
        raise KeyError(f"world version {version!r} not in registry")
    if LEVELS.index(_level(v)) < LEVELS.index(min_level):
        raise ValueError(f"world version {version} is {_level(v)}, below {min_level}")
    _check_files(v)
    return v["world_dir"]


def diff_programs(old_src, new_src):
    """Coarse change record: which source lines were added/removed."""
    a, b = set(old_src.splitlines()), set(new_src.splitlines())
    return {"removed": sorted(a - b)[:50], "added": sorted(b - a)[:50]}
