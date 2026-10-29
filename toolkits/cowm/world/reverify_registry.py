"""Re-verify schema-1 registry entries and register verified copies as NEW versions (Step 4, CPU).

Old entries are never edited. For each old entry:
  1. world_dir and world.json exist and world.json sha256 == the registered world_json_sha256
  2. the program next to it (<round>/world_program.py) exists and its sha256 == world.json program_sha256
  3. the verifier (check_program) is re-run on that program into a fresh directory
  4. a new version is registered for the fresh directory with parent = old version. Its report is
     ok only if the fresh verifier passed AND the old entry recorded no errors (a recorded failure, e.g. a
     structural-check failure that schema 1 still published, is carried over, never washed out).
Anything failing 1-2 is reported as unverifiable and nothing is registered for it.
usage: COWM_REGISTRY=<copy> python reverify_registry.py <registry_copy> <out_root> [--dry-run]
"""
import hashlib
import json
import os
import sys

from cowm.world import registry as reg
from cowm.world.check_program import check


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


SEMANTIC_KEYS = ("program_sha256", "table", "camera", "objects", "materials", "mechanisms", "params")


def file_shas(d):
    out = {}
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            if f == "check_report.json":   # verifier output, not a world asset
                continue
            p = os.path.join(root, f)
            out[os.path.relpath(p, d)] = sha(p)
    return out


def rebuild_diff(old_dir, new_dir):
    """What differs between the registered world and its rebuild: semantic world.json fields (sdk_version is
    reported separately, it changes by design), and every file (XML / CoACD meshes) by sha256."""
    a = json.load(open(os.path.join(old_dir, "world.json")))
    b = json.load(open(os.path.join(new_dir, "world.json")))
    fa, fb = file_shas(old_dir), file_shas(new_dir)
    common = set(fa) & set(fb)
    diff = {"semantic_fields_differ": [k for k in SEMANTIC_KEYS if a.get(k) != b.get(k)],
            "sdk_version": [a.get("sdk_version"), b.get("sdk_version")],
            "files_only_old": sorted(set(fa) - set(fb)), "files_only_new": sorted(set(fb) - set(fa)),
            "files_differ": sorted(f for f in common if fa[f] != fb[f] and f != "world.json"),
            "files_same": sum(1 for f in common if fa[f] == fb[f])}
    diff["semantic_equal"] = not diff["semantic_fields_differ"]
    diff["assets_equal"] = not (diff["files_only_old"] or diff["files_only_new"] or diff["files_differ"])
    return diff


def main():
    path, out_root = sys.argv[1], sys.argv[2]
    dry = "--dry-run" in sys.argv
    old = [v for v in reg._load(path)["versions"] if v.get("tree_sha256") is None]
    rows = []
    for v in old:
        r = {"old": v["version"], "old_status": v["status"], "old_errors": len(v.get("errors", []))}
        wj = os.path.join(v["world_dir"], "world.json")
        if not os.path.isfile(wj) or sha(wj) != v["world_json_sha256"]:
            r["result"] = "unverifiable: world.json missing or changed"
            rows.append(r)
            continue
        prog = os.path.join(os.path.dirname(v["world_dir"]), "world_program.py")
        psha = json.load(open(wj)).get("program_sha256")
        if not os.path.isfile(prog) or hashlib.sha256(open(prog).read().encode()).hexdigest() != psha:
            r["result"] = "unverifiable: program missing or does not match program_sha256"
            rows.append(r)
            continue
        if dry:
            r["result"] = "verifiable (dry run)"
            rows.append(r)
            continue
        fresh = os.path.join(out_root, v["version"])
        if os.path.exists(fresh):
            r["result"] = f"refused: {fresh} exists (each rebuild needs its own directory)"
            rows.append(r)
            continue
        rep = check(prog, fresh)
        wdir = os.path.join(fresh, "world") if os.path.isdir(os.path.join(fresh, "world")) else fresh
        if not os.path.isfile(os.path.join(wdir, "world.json")):
            r.update(result="rebuild failed: no world.json", verifier_ok=False, errors=rep.get("errors", [])[:2])
            rows.append(r)
            continue
        diff = rebuild_diff(v["world_dir"], wdir)
        r["rebuild_diff"] = {k: diff[k] for k in ("semantic_equal", "semantic_fields_differ", "assets_equal",
                                                  "files_same", "files_differ", "files_only_old",
                                                  "files_only_new", "sdk_version")}
        if not diff["semantic_equal"]:
            r["result"] = "not registered: rebuild is semantically a different world"
            rows.append(r)
            continue
        ok = bool(rep.get("ok")) and not v.get("errors")
        rep2 = dict(rep, ok=ok, errors=list(rep.get("errors", [])) + [f"carried from {v['version']}: {e}"
                                                                      for e in v.get("errors", [])])
        nv = reg.register(wdir, parent=v["version"], source=f"reverify of {v['version']} ({v.get('source', '')})",
                          evidence_ids=v.get("evidence_ids", []),
                          changed={"reverify": True, "rebuild_diff": r["rebuild_diff"]}, report=rep2, path=path)
        r.update(result="registered", new=nv, verifier_ok=bool(rep.get("ok")),
                 new_status=next(x["status"] for x in reg._load(path)["versions"] if x["version"] == nv))
        rows.append(r)
    for r in rows:
        print(json.dumps(r))
    print("REVERIFY_DONE", len(rows), "old entries")


if __name__ == "__main__":
    main()
