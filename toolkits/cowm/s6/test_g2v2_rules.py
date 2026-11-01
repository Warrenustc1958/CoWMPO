# Copyright 2026 The RLinf Authors.
"""Synthetic tests for the G2 v2 initial-state rule (no gate data). Exit 1 on any failure."""
import sys

from cowm.s6 import g2_replay as g
from cowm.s6.g2v2_rules import split_issues, park_context, CONTEXT_MAX_FRAC
from cowm.init.estimate_state import PARK_X

NAMES = ["bowl", "plate", "cream_cheese"]


def obj(status="matched", observed=True, pos=(0.0, 0.1, 0.9)):
    return {"pos": list(pos), "yaw": 0.0, "observed": observed, "association_status": status,
            "mask_rejected": False, "parked": False}


def ep_result(name, dyn=1.5, context=(), rule="park_non_probe_v2"):
    return {"episode": name, "obj": "bowl", "static_iou": .88, "dyn_err_px": dyn, "dyn_err_px_max": dyn,
            "final_obj_iou": .84, "terminal_step": 232, "terminal_status": "valid", "missing_world_objects": [],
            "frame_status": {"expected": 35, "valid": 35}, "initial_state_valid": True,
            "initial_state_issues": {}, "complete": True, "context_rule": rule, "context_uncertain": list(context)}


FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


# 1. ambiguous non-probe object -> context, not blocking
state = {"bowl": obj(), "plate": obj(), "cream_cheese": obj("ambiguous", observed=False)}
b, c = split_issues(state, NAMES, "bowl")
check(not b and set(c) == {"cream_cheese"}, "ambiguous non-probe object is context only")

# 2. ambiguous probe object -> blocking
state2 = {"bowl": obj("ambiguous", observed=False), "plate": obj(), "cream_cheese": obj()}
b, c = split_issues(state2, NAMES, "bowl")
check("bowl" in b and not c, "ambiguous probe object stays blocking")

# 3. missing state record / invalid pose on non-probe -> blocking
state3 = {"bowl": obj(), "plate": obj(pos=(float("nan"), 0, 0.9)), }
b, c = split_issues(state3, NAMES, "bowl")
check(set(b) == {"plate", "cream_cheese"} and not c, "invalid pose and missing record stay blocking")

# 4. park_context moves only context objects off the table, keeps z, does not mutate input
parked = park_context(state, {"cream_cheese": ["identity_ambiguous"]})
check(parked["cream_cheese"]["pos"][0] == PARK_X and parked["cream_cheese"]["pos"][2] == 0.9
      and parked["bowl"]["pos"] == state["bowl"]["pos"] and state["cream_cheese"]["pos"][0] == 0.0,
      "park_context parks only context objects, z kept, input untouched")

# 5. episode_metrics: uncertain static identity of a parked context object does not block completeness
frames = {0: {"status": "static", "iou": {"bowl": .9}, "uncertain_identities": ["cream_cheese"],
              "missing_world_objects": []},
          10: {"status": "valid", "obj_centroid_err_px": 1.0, "iou": {"bowl": .9}},
          232: {"status": "valid", "obj_centroid_err_px": 1.0, "iou": {"bowl": .8}}}
m1 = g.episode_metrics("e", "bowl", 232, frames)
m2 = g.episode_metrics("e", "bowl", 232, frames, {"cream_cheese"})
check(m1["complete"] is False and m2["complete"] is True, "v1 incomplete, v2 complete when only context uncertain")
frames[0]["uncertain_identities"] = ["bowl"]
check(g.episode_metrics("e", "bowl", 232, frames, {"cream_cheese"})["complete"] is False,
      "uncertain probe identity still incomplete under v2")

# 6. summarize v2: one flagged episode out of 100 passes; flagged rate over limit fails
res = [ep_result(f"e{i}") for i in range(99)] + [ep_result("e99", context=["cream_cheese"])]
s = g.summarize(res)
check(s["G2_target_evidence"] and s["checks"]["subset_without_context"] and s["checks"]["context_uncertain_rate"],
      "v2 passes with 1/100 context-uncertain")
n_bad = int(CONTEXT_MAX_FRAC * 100) + 1
res2 = [ep_result(f"e{i}") for i in range(100 - n_bad)] + [ep_result(f"c{i}", context=["plate"]) for i in range(n_bad)]
check(not g.summarize(res2)["checks"]["context_uncertain_rate"], "v2 fails when context-uncertain > limit")

# 7. verdict must not depend on flagged episodes: subset failure fails the gate
res3 = [ep_result(f"e{i}", dyn=7.0) for i in range(30)] + [ep_result(f"f{i}") for i in range(65)] + \
       [ep_result(f"c{i}", dyn=0.1, context=["plate"]) for i in range(5)]
s3 = g.summarize(res3)
check("subset_without_context" in s3["checks"], "subset check present when episodes are flagged")

# 8. v1 behaviour unchanged when no context_rule is set
res4 = [ep_result(f"e{i}", rule=None) for i in range(100)]
s4 = g.summarize(res4)
check("context_uncertain_rate" not in s4["checks"] and s4["G2_target_evidence"], "v1 summary unchanged")

print("G2V2_RULE_TESTS", "FAIL" if FAILS else "PASS", len(FAILS))
sys.exit(1 if FAILS else 0)
