# Copyright 2026 The RLinf Authors.
"""Dense public-RGB plan for the 6 rejected v24 acceptance positions (v25 fit evidence). Builder side, no GT.

The v24 Target result has been seen, so these positions are fit evidence now. Actions are copied unchanged from
rack_accept_plan_v24; only the exported frames are densified around first contact (every step in [190, 260],
every 10 steps elsewhere), exactly like rack_accept_dense_plan_v23. The Target run exports RGB only; candidates are
scored on public tracks of these frames, never on Target numbers.
usage: python plan_rack_accept_dense_v25.py
"""
import hashlib
import json
from pathlib import Path

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
SRC = R / "rack_accept_plan_v24"
OUT = R / "rack_accept_dense_plan_v25"
WINDOW = (190, 260)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    keys = sorted(json.loads((SRC / "freeze.json").read_text())["grid_mm"])
    if len(keys) != 6:
        raise ValueError(keys)
    OUT.mkdir(exist_ok=False)
    for key in keys:
        src = SRC / key / "probe.json"
        p = json.loads(src.read_text())
        n = len(p["actions"])
        frames = sorted(set(range(0, n + 1, 10)) | set(range(WINDOW[0], WINDOW[1] + 1)))
        p.update(frames=frames, dense_window=list(WINDOW), derived_from=f"{src} (actions unchanged, frames densified)",
                 purpose="fit evidence (v24 Target result seen); dense public RGB around first contact")
        (OUT / key).mkdir()
        (OUT / key / "probe.json").write_text(json.dumps(p, indent=1))
        print("DENSE_PLAN", key, len(frames), sha(src)[:12], flush=True)
    (OUT / "freeze.json").write_text(json.dumps({
        "keys": keys, "window": WINDOW, "source_freeze_sha256": sha(SRC / "freeze.json"), "script_sha256": sha(__file__),
        "GT_inputs": False, "data_roles": "fit evidence (v24 rejected; Target first contact seen)"}, indent=1))
    print("DENSE_PLAN_V25_DONE", flush=True)


if __name__ == "__main__":
    main()
