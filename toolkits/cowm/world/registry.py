"""CoWM-PO world version registry (plan §9.7): immutable, append-only world versions.

Every compiled world is registered with its parent version, what changed, which evidence it saw, its
verifier report and cost. A version is PUBLISHED only if G0 and G1 pass; only published versions may be
loaded for rollouts, and one GRPO group always uses a single (world_version, rm_version) pair.
"""
import fcntl
import hashlib
import json
import os
import time

REGISTRY = os.environ.get("COWM_REGISTRY", "/data_cf/cowmpo/worlds/world_registry.json")


def _load(path):
    if not os.path.isfile(path):
        return {"schema": "cowm-registry-1", "versions": []}
    with open(path) as f:
        return json.load(f)


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def register(world_dir, parent=None, source="", evidence_ids=(), changed=None, report=None, cost=None,
             path=REGISTRY):
    """Append one version. Returns its version id (e.g. 'w0007')."""
    world_json = os.path.join(world_dir, "world.json")
    world = json.load(open(world_json)) if os.path.isfile(world_json) else {}
    gates = (report or {}).get("gates", {})
    published = bool(gates.get("G0_build") and gates.get("G1_self_consistent"))
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
            "sdk_version": world.get("sdk_version"), "evidence_ids": list(evidence_ids),
            "changed": changed or {}, "gates": gates, "errors": (report or {}).get("errors", []),
            "cost": cost or {}, "status": "published" if published else "rejected"})
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(reg, f, indent=2)
        os.replace(tmp, path)
    return vid


def resolve(version, path=REGISTRY, require_published=True):
    """Return the world_dir of a version after checking it is unchanged since registration."""
    reg = _load(path)
    v = next((x for x in reg["versions"] if x["version"] == version), None)
    if v is None:
        raise KeyError(f"world version {version!r} not in registry")
    if require_published and v["status"] != "published":
        raise ValueError(f"world version {version} is {v['status']}, not published")
    wj = os.path.join(v["world_dir"], "world.json")
    if _sha_file(wj) != v["world_json_sha256"]:
        raise ValueError(f"world {version} was modified after registration (world.json hash mismatch)")
    return v["world_dir"]


def diff_programs(old_src, new_src):
    """Coarse change record: which source lines were added/removed."""
    a, b = set(old_src.splitlines()), set(new_src.splitlines())
    return {"removed": sorted(a - b)[:50], "added": sorted(b - a)[:50]}
