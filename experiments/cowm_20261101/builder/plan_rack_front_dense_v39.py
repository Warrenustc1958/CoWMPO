# Copyright 2026 The RLinf Authors.
"""Dense public-RGB re-export of the three existing front-row vertical probes vert_x-080_* (already fit evidence since
v3/v12) so the deceleration onset can test the near-end extent (S5). Actions copied unchanged; frames densified over
steps 190-260 like the other dense plans. Builder side, no GT.
usage: python plan_rack_front_dense_v39.py
"""
import hashlib
import json
from pathlib import Path

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
SRC = R / "rack_vertical_preflight_dev_v3"
OUT = R / "rack_front_dense_plan_v39"
KEYS = ["vert_x-080_y-080", "vert_x-080_y+000", "vert_x-080_y+080"]
WINDOW = (190, 260)


def main():
    OUT.mkdir(exist_ok=False)
    for k in KEYS:
        src = SRC / k / "probe.json"
        p = json.loads(src.read_text())
        p["frames"] = sorted(set(range(0, len(p["actions"]) + 1, 10)) | set(range(WINDOW[0], WINDOW[1] + 1)))
        p.update(dense_window=list(WINDOW), derived_from=f"{src} (actions unchanged, frames densified)",
                 purpose="fit evidence: front-row near-end dense public RGB (S5 extent)")
        (OUT / k).mkdir()
        (OUT / k / "probe.json").write_text(json.dumps(p, indent=1))
        print("FRONT_PLAN", k, len(p["frames"]), hashlib.sha256(src.read_bytes()).hexdigest()[:12], flush=True)
    (OUT / "freeze.json").write_text(json.dumps({"keys": KEYS, "window": WINDOW, "GT_inputs": False,
                                                 "data_roles": "fit evidence (existing v3 front-row probes)"}, indent=1))
    print("FRONT_PLAN_V39_DONE", flush=True)


if __name__ == "__main__":
    main()
