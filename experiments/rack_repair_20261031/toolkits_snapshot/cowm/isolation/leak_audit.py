"""S1 leak audit (Evaluator side, never run inside the Builder).

Flags a Builder-produced MJCF that reuses LIBERO assets or copies LIBERO geometry:
  (a) any referenced mesh/texture file whose sha256 matches a LIBERO asset file
  (b) primitive collision geoms whose (type, size, pos, quat) signature matches a LIBERO geom
Physical parameters (friction, solref, density) are not used: default values coincide legitimately.
"""
import argparse
import glob
import hashlib
import json
import os
import xml.etree.ElementTree as ET

ASSET_EXT = (".msh", ".obj", ".stl", ".png", ".jpg", ".jpeg")
PRIMS = ("box", "capsule", "cylinder", "ellipsoid", "sphere")


def sha(path, cache={}):
    if path not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[path] = h.hexdigest()
    return cache[path]


def geom_sigs(xml_path):
    sigs = set()
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return sigs
    for g in root.iter("geom"):
        t = g.get("type", "sphere")
        if t not in PRIMS or g.get("size") is None:
            continue
        r = lambda s, d: tuple(round(float(x), 4) for x in (s or d).split())  # noqa: E731
        sigs.add((t, r(g.get("size"), ""), r(g.get("pos"), "0 0 0"), r(g.get("quat"), "1 0 0 0")))
    return sigs


def referenced_files(xml_path):
    root = ET.parse(xml_path).getroot()
    base = os.path.dirname(os.path.abspath(xml_path))
    comp = root.find("compiler")
    meshdir = os.path.join(base, comp.get("meshdir", "")) if comp is not None else base
    texdir = os.path.join(base, comp.get("texturedir", "")) if comp is not None else base
    out = []
    for tag, d in (("mesh", meshdir), ("texture", texdir), ("hfield", base)):
        for e in root.iter(tag):
            f = e.get("file")
            if f:
                p = f if os.path.isabs(f) else os.path.join(d, f)
                if os.path.isfile(p):
                    out.append(p)
    return out


def _scan(root):
    hashes, sigs = {}, set()
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith(ASSET_EXT) and os.path.isfile(p):
            hashes[sha(p)] = os.path.relpath(p, root)
        elif p.endswith(".xml") and os.path.isfile(p):
            sigs |= geom_sigs(p)
    return hashes, sigs


def build_index(libero_assets, allowed_roots=()):
    """LIBERO index minus anything that also exists in an allowed shared-prior library
    (e.g. robosuite generic textures, which LIBERO itself reuses)."""
    hashes, sigs = _scan(libero_assets)
    shared = 0
    for r in allowed_roots:
        ah, asig = _scan(r)
        for h in ah:
            if h in hashes:
                del hashes[h]
                shared += 1
        sigs -= asig
    return hashes, sigs, shared


def audit(candidate_xmls, hashes, sigs, sig_tol=0.05):
    report = {"files": [], "asset_hash_hits": [], "geom_sig_hits": 0, "geom_sig_total": 0}
    for x in candidate_xmls:
        for f in referenced_files(x):
            h = sha(f)
            if h in hashes:
                report["asset_hash_hits"].append({"file": f, "libero": hashes[h]})
        s = geom_sigs(x)
        report["geom_sig_total"] += len(s)
        report["geom_sig_hits"] += len(s & sigs)
        report["files"].append(x)
    tot = max(report["geom_sig_total"], 1)
    report["geom_sig_ratio"] = report["geom_sig_hits"] / tot
    report["leak"] = bool(report["asset_hash_hits"]) or report["geom_sig_ratio"] > sig_tol
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--libero-assets", required=True)
    p.add_argument("--candidates", nargs="+", required=True, help="MJCF files or globs")
    p.add_argument("--allow-assets", nargs="*", default=[],
                   help="shared-prior asset roots (e.g. robosuite/models/assets) excluded from the index")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    hashes, sigs, shared = build_index(args.libero_assets, args.allow_assets)
    xmls = sorted({f for c in args.candidates for f in glob.glob(c, recursive=True)})
    rep = audit(xmls, hashes, sigs)
    rep["index"] = {"libero_asset_files": len(hashes), "libero_geom_sigs": len(sigs),
                    "excluded_shared_with_allowed": shared}
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"candidates={len(xmls)} asset_hash_hits={len(rep['asset_hash_hits'])} "
          f"geom_sig={rep['geom_sig_hits']}/{rep['geom_sig_total']} ({rep['geom_sig_ratio']:.3f}) "
          f"LEAK={rep['leak']}")


if __name__ == "__main__":
    main()
