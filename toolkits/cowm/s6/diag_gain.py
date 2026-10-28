"""S6 debug: waypoint tracking error vs P-gain and action cap (code world, rack parked)."""
import json, sys
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
world, state_f = sys.argv[1], sys.argv[2]
sc = Scene(world)
st = json.load(open(state_f))["objects"]
bx, by = st["bowl"]["pos"][:2]
top = sc.world["table"]["height"]
for gain, cap in [(10, 1.0), (4, 0.5), (2, 0.3), (6, 0.3)]:
    sc.reset(); apply_state(sc, st)
    errs = []
    for tgt in ([bx, by + 0.1, top + 0.11], [bx, by + 0.1, top + 0.03], [bx, by + 0.025, top + 0.03]):
        for _ in range(60):
            a = np.zeros(7); a[6] = 1.0
            a[:3] = np.clip(gain * (np.asarray(tgt) - sc.env.eef_pos()), -cap, cap)
            sc.env.step(a)
        errs.append(round(float(np.linalg.norm(sc.env.eef_pos() - np.asarray(tgt))) * 100, 1))
    print("gain", gain, "cap", cap, "final err cm per waypoint", errs, "bowl", np.round(sc.env.object_pos("bowl")[:2], 3))
