# Copyright 2026 The RLinf Authors.
"""Driver for v61: 3 shrinking rounds (scale .5, .25, .12) of 96 samples, each centred on the previous round's best
(selection order: final correct, max onset error, frame errors). Builder side, public probes only."""
import json, os, subprocess, sys
from pathlib import Path
R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
O = Path(__file__).resolve().parent
B = "/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh"
c = json.loads((R / "stove_detent_v63_r2/freeze.json").read_text())["candidates"]
center = next(x for x in c if x["name"] == "r2_097")["params"] + [1.]
hist = []
for r, scale in enumerate((.6, .3), 1):
    dest = f"stove_detent_v66_r{r}"
    env = dict(os.environ, V61_DEST=dest, V61_SEED=str(660 + r), V61_CENTER=json.dumps(center), V61_SCALE=str(scale))
    for arg in (["--prepare"], []):
        p = subprocess.run([B, "python", str(O / "stove_detent_v66.py"), *arg], env=env, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    s = json.loads((R / dest / "selection.json").read_text())
    top = s["top"][0]
    cc = json.loads((R / dest / "freeze.json").read_text())["candidates"]
    center = next(x for x in cc if x["name"] == top["cand"])["params"]
    flip = next(x for x in cc if x["name"] == top["cand"])["axis_flip"]
    print("TOP_AXIS_FLIP", flip, flush=True)
    hist.append({"round": r, "scale": scale, "top": top, "center_out": center, "loo_final": s["loo_final_correct"],
                 "loo_onset": s["loo_onset_errors"], "n_eligible": s["n_eligible"]})
    print("ROUND", r, json.dumps({k: top[k] for k in ("cand", "thr", "final", "max_err", "frame_err", "eligible")}),
          "LOO", s["loo_final_correct"], s["loo_onset_errors"], "eligible", s["n_eligible"], flush=True)
(R / "stove_detent_v66_history.json").write_text(json.dumps(hist, indent=1))
print("V66_DONE", flush=True)
