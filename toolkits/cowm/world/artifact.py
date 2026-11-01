# Copyright 2026 The RLinf Authors.
"""Portable content checks for frozen compiled worlds, without simulator or registry dependencies."""
import hashlib
from pathlib import Path

from cowm.init.observation import sha_file


def tree_sha(world_dir):
    """Hash relative POSIX paths and file contents using the Linux registry's ordering."""
    root = Path(world_dir)
    if not root.is_dir() or not (root / "world.json").is_file():
        raise ValueError(f"compiled world missing: {root}")
    h = hashlib.sha256()

    def visit(directory):
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
        for path in entries:
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                h.update(rel.encode() + b"\0" + sha_file(path).encode() + b"\n")
        for path in entries:
            if path.is_dir() and path.name != "__pycache__":
                visit(path)

    visit(root)
    return h.hexdigest()


def verify_tree(world_dir, expected):
    """Reject absent hashes, changed XML/meshes and a substituted compiled world."""
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("expected frozen world directory SHA256 required")
    actual = tree_sha(world_dir)
    if actual != expected:
        raise ValueError(f"world directory hash mismatch: expected {expected}, received {actual}")
    return actual
