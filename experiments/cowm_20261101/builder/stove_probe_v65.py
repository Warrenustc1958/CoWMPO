# Copyright 2026 The RLinf Authors.
"""Stove fit probes v65 (Builder side, no GT): same branching as Codex stove_active_contrast (policy prefix up to
observed_onset-12, then 18 wrist-twist steps, lift, retract), changing ONLY twist direction and magnitude, at
magnitudes never used before (.05, .2; earlier probes used .1). t7_i0: axis {0,2} x sign x {.05,.2}; t7_i1/t7_i2:
axis 0 x sign x .2. 12 probes, fit evidence (direction sensitivity of the knob)."""
import json
from pathlib import Path
from builder_batch import OUT, sha, write

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
DEST = R / "stove_probe_v65"
PRIOR = OUT / "stove_collision_modes_v1"
SPEC = [(0, a, s, m) for a in (0, 2) for s in (-1, 1) for m in (.05, .2)] + [(t, 0, s, .2) for t in (1, 2) for s in (-1, 1)]


def main():
    DEST.mkdir(exist_ok=False)
    keys = []
    for trial, axis, sign, mag in SPEC:
        source = PRIOR / f"policy_libero_goal_t7_i{trial}_query.json"
        q = json.loads(source.read_text())
        prefix = max(0, q["observed_onset"] - 12)
        grip = q["actions"][prefix - 1][-1]
        wrist = [0.] * 7
        wrist[axis + 3] = sign * mag
        wrist[-1] = grip
        acts = q["actions"][:prefix] + [wrist] * 18 + [[0, 0, .3, 0, 0, 0, -1.]] * 20 + [[0, 0, 0, 0, 0, 0, -1.]] * 20
        key = f"{q['episode']}__v65_a{axis}_{'p' if sign == 1 else 'm'}{int(mag * 100):02d}"
        frames = sorted(set(range(0, len(acts) + 1, 2)) | {prefix, len(acts)})
        write(DEST / f"{key}_query.json", {**q, "key": key, "actions": acts, "frames": frames, "prefix_steps": prefix,
                                           "axis": axis, "sign": sign, "magnitude": mag,
                                           "purpose": "fit evidence: twist direction/magnitude only",
                                           "source_query_sha256": sha(source)})
        keys.append(key)
    write(DEST / "freeze.json", {"queries": keys, "query_sha256": {k: sha(DEST / f"{k}_query.json") for k in keys},
                                 "spec": SPEC, "script_sha256": sha(__file__), "GT_inputs": False,
                                 "data_roles": "fit evidence"})
    print("V65_PROBES", keys, flush=True)


if __name__ == "__main__":
    main()
