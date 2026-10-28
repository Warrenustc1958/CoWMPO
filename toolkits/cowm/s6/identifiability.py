"""S6 identifiability check (Builder side, simulation only): which dynamics parameters does a probe see?

For each candidate parameter, replay the planned probe actions in the code world at {lo, init, hi} of the
declared prior range (others at init) and record the probe object's final image centroid. A parameter is
called identifiable by this probe iff the centroid spread over its range exceeds MIN_PX (perception noise).
Only identifiable parameters are fitted; the rest keep their priors (reported, not hidden).
"""
import argparse
import json
import os

import numpy as np

from cowm.init.estimate_state import Scene
from cowm.s6.params import apply_params
from cowm.s6.plan_probe import apply_state

MIN_PX = 2.0


def final_centroid(scene, state, probe, values):
    apply_params(scene, values)
    scene.reset()
    apply_state(scene, state)
    for act in probe["actions"]:
        scene.env.step(np.asarray(act))
    k = scene.names.index(probe["obj"])
    ys, xs = np.nonzero(scene.seg() == k)
    disp = scene.env.object_pos(probe["obj"])[:2]
    return (np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])), disp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--params", default="tabletop.friction,ceramic.density,ceramic.friction")
    p.add_argument("--episodes", default="")
    a = p.parse_args()
    scene = Scene(a.world)
    spec = scene.world["params"]
    names = a.params.split(",")
    eps = a.episodes.split(",") if a.episodes else sorted(os.listdir(a.probes))
    rep = {}
    for n in names:
        lo, hi = spec[n]["bounds"]
        grid = [lo, spec[n]["init"], hi]
        spreads, disps = [], []
        for ep in eps:
            state = json.load(open(os.path.join(a.state, ep, "state.json")))["objects"]
            probe = json.load(open(os.path.join(a.probes, ep, "probe.json")))
            cs, ds = zip(*[final_centroid(scene, state, probe, {n: v}) for v in grid])
            cs = np.array(cs)
            spreads.append(float(np.nanmax(np.linalg.norm(cs - cs[1], axis=1))))
            disps.append([float(np.linalg.norm(d - ds[1]) * 100) for d in ds])
        rep[n] = {"grid": grid, "centroid_spread_px_mean": float(np.mean(spreads)),
                  "centroid_spread_px_per_ep": spreads, "final_pos_change_cm": disps,
                  "identifiable": bool(np.mean(spreads) > MIN_PX)}
        print("IDENT", n, "grid", [round(g, 3) for g in grid], "spread px", round(float(np.mean(spreads)), 2),
              "identifiable", rep[n]["identifiable"], flush=True)
    apply_params(scene, {})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
