"""Episode data roles (profile splits_goal_v1): the single place that decides fit / dev / gate / test.

Role of an episode (task_id, trial_id):
  1. any episode listed in exposed_to_agent is dev forever (rule: shown to the Agent -> dev);
  2. otherwise the role of its trial_id in splits["roles"].
Consumers must ask for the roles they are allowed to read; anything else raises RoleError, so a gate or test
episode can never be fitted on by accident (audit 10.29: s6_identify_v2 used the "_i2" name suffix).
"""
import json
import os
import re

SPLITS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles", "splits_goal_v1.json")
_EP = re.compile(r"^libero_goal_t(\d+)_i(\d+)$")


class RoleError(ValueError):
    pass


def load(path=SPLITS):
    d = json.load(open(path))
    by_trial = {int(t): r for r, ts in d["roles"].items() for t in ts}
    return {"profile": d.get("profile"), "by_trial": by_trial, "exposed": set(d.get("exposed_to_agent", [])),
            "path": os.path.abspath(path)}


def parse(ep):
    m = _EP.match(ep)
    if not m:
        raise RoleError(f"episode name {ep!r} is not libero_goal_t<task>_i<trial>")
    return int(m.group(1)), int(m.group(2))


def role(ep, splits=None):
    s = splits or load()
    if ep in s["exposed"]:
        return "dev"
    _, trial = parse(ep)
    if trial not in s["by_trial"]:
        raise RoleError(f"{ep}: trial {trial} has no role in {s['path']}")
    return s["by_trial"][trial]


def roles_of(eps, splits=None):
    s = splits or load()
    return {e: role(e, s) for e in eps}


def select(eps, wanted, splits=None):
    """Episodes of eps whose role is in wanted (str or iterable)."""
    wanted = {wanted} if isinstance(wanted, str) else set(wanted)
    r = roles_of(eps, splits)
    return [e for e in eps if r[e] in wanted]


def require(eps, allowed, purpose, splits=None):
    """Raise RoleError if any episode has a role outside allowed (e.g. gate/test handed to a fitter)."""
    allowed = {allowed} if isinstance(allowed, str) else set(allowed)
    bad = {e: r for e, r in roles_of(eps, splits).items() if r not in allowed}
    if bad:
        raise RoleError(f"{purpose}: episodes with role outside {sorted(allowed)}: {bad}")
    return eps
