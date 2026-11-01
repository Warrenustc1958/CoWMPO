# Copyright 2026 The RLinf Authors.
"""G2 v2 initial-state rule (declared 2026-11-01 before any v2 scoring; see protocol_v2.json).

v1 invalidated an episode when ANY world object lacked a matched, observed state, and gate_robot.py then
skipped robot estimation, so one ambiguous non-probe object (gate t0_i35 cream_cheese) removed the whole
episode. v2 splits the issues:

  blocking  probe object not matched/observed/valid, robot/episode/world problems, any object with no
            state record or a non-finite pose  -> episode invalid (unchanged from v1)
  context   a NON-probe object that is ambiguous / unobserved / mask-rejected / parked
            -> object is parked off the table (no contact possible), episode replayed and flagged
               context_uncertain; G2 v2 must pass on all episodes AND on the subset without them, and
               context_uncertain episodes may be at most CONTEXT_MAX_FRAC of the population.

Perception thresholds (association MARGIN etc.) are NOT changed.
"""
import copy

from cowm.init.estimate_state import PARK_X
from cowm.init.observation import state_issues

RULE = "park_non_probe_v2"
CONTEXT_MAX_FRAC = 0.1
CONTEXT_REASONS = {"identity_ambiguous", "identity_missing", "identity_unverified", "unobserved",
                   "mask_rejected", "parked"}


def split_issues(objects, names, probe_obj):
    """Return (blocking, context) issue dicts for one initial state."""
    issues = state_issues(objects, names)
    blocking, context = {}, {}
    for name, reasons in issues.items():
        if name != probe_obj and set(reasons) <= CONTEXT_REASONS:
            context[name] = reasons
        else:
            blocking[name] = reasons
    return blocking, context


def park_context(objects, context):
    """Copy of the state with every context object parked off the table (same xy rule as S5, z kept)."""
    out = copy.deepcopy(objects)
    for k, name in enumerate(sorted(context)):
        s = out[name]
        s["pos"] = [PARK_X, 0.6 + 0.3 * k, float(s["pos"][2])]
        s["parked"] = True
        s["context_parked_v2"] = True
    return out
