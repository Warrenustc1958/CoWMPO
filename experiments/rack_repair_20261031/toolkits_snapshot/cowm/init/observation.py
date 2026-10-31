# Copyright 2026 The RLinf Authors.
"""Contracts between identity association, pose estimation and simulation."""
import hashlib
import json
from pathlib import Path

import numpy as np

STATUSES = {"matched", "ambiguous", "missing"}


def sha_file(path):
    """Return the content hash of a file without loading it into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_masks(names, masks, association=None, allow_legacy=False):
    """Return usable masks and explicit identity status for every world object.

    Tentative masks remain in the evidence artifact, but never become pose observations.
    Legacy input is an explicit diagnostic mode and is marked unverified downstream.
    """
    names = list(names)
    if association is None:
        if not allow_legacy:
            raise ValueError("association.json required; legacy masks need explicit diagnostic mode")
        records = {n: {"status": "unverified"} for n in names}
    else:
        records = association.get("objects", {})
        if set(records) != set(names):
            raise ValueError("association object set must equal the world object set")
        for n, record in records.items():
            if record.get("status") not in STATUSES:
                raise ValueError(f"{n}: invalid association status {record.get('status')!r}")
    usable = {}
    status = {}
    for n in names:
        s = records[n]["status"]
        mk = masks.get(n)
        if s == "matched" and (mk is None or not np.asarray(mk).any()):
            raise ValueError(f"{n}: matched association requires a nonempty mask")
        if mk is not None and (np.asarray(mk).ndim != 2 or np.asarray(mk).dtype != np.bool_):
            raise ValueError(f"{n}: expected a two-dimensional boolean mask")
        if mk is not None and np.asarray(mk).any() and (s == "matched" or allow_legacy and s == "unverified"):
            usable[n] = mk
        status[n] = s
    return usable, status


def load_association(directory, episode):
    """Load metadata bound to the exact masks.npz consumed by pose estimation."""
    directory = Path(directory)
    path = directory / "association.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != 1 or record.get("episode") != episode:
        raise ValueError(f"{episode}: invalid association schema or episode")
    if record.get("masks_sha256") != sha_file(directory / "masks.npz"):
        raise ValueError(f"{episode}: masks differ from association metadata")
    return record


def state_issues(objects, names):
    """Return reasons a state cannot initialize a scored simulation; never infer certainty."""
    issues = {}
    for n in names:
        s = objects.get(n)
        reasons = []
        if not isinstance(s, dict):
            issues[n] = ["missing_state"]
            continue
        if s.get("association_status") != "matched":
            reasons.append("identity_" + str(s.get("association_status", "unverified")))
        if s.get("observed") is not True:
            reasons.append("unobserved")
        if s.get("mask_rejected"):
            reasons.append("mask_rejected")
        if s.get("parked"):
            reasons.append("parked")
        try:
            pos = np.asarray(s.get("pos"), dtype=float)
            yaw = float(s.get("yaw"))
            valid = pos.shape == (3,) and np.isfinite(pos).all() and np.isfinite(yaw)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            reasons.append("invalid_pose")
        if reasons:
            issues[n] = reasons
    return issues
