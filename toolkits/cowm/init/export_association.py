# Copyright 2026 The RLinf Authors.
"""Export v3 association and bound tentative masks from a frozen candidate pool; no models or GT."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from cowm.init.associate import associate
from cowm.init.observation import sha_file


def export_episode(pool, out, names, episode):
    """Create a new, self-contained episode artifact; refuse overwriting old evidence."""
    source = Path(pool) / episode
    output = Path(out) / episode
    audit_path = source / "audit.json"
    mask_path = source / "candidate_masks.npz"
    audit_hash, mask_hash = sha_file(audit_path), sha_file(mask_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("episode") != episode:
        raise ValueError(f"{episode}: candidate audit episode mismatch")
    with np.load(mask_path, allow_pickle=False) as archive:
        masks = {c["cid"]: archive[f"c{c['cid']}"] for c in audit["candidates"]}
    result = associate(names, audit["candidates"], masks)
    if (sha_file(audit_path), sha_file(mask_path)) != (audit_hash, mask_hash):
        raise ValueError(f"{episode}: candidate pool changed while reading")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "masks.npz", **{
        n: masks[r["cid"]] for n, r in result["objects"].items() if r["cid"] is not None})
    record = {"schema": 1, "episode": episode, "algorithm": "associate_v3_containment",
              "objects": result["objects"], "masks_sha256": sha_file(output / "masks.npz"),
              "source": {"audit_sha256": audit_hash, "candidate_masks_sha256": mask_hash}}
    (output / "association.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "out", "world", "episodes"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    from cowm import roles
    episodes = args.episodes.split(",")
    base_episodes = [re.sub(r"__f\d+$", "", ep) for ep in episodes]
    roles.require(base_episodes, {"dev", "fit"}, "association repair export")
    if len(episodes) != len(set(episodes)):
        raise ValueError("duplicate requested episode")
    world = json.loads((Path(args.world) / "world.json").read_text(encoding="utf-8"))
    names = [o["name"] for o in world["objects"]]
    for ep in episodes:
        export_episode(args.pool, args.out, names, ep)


if __name__ == "__main__":
    main()
