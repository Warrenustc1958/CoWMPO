"""S6 debug: eef vs waypoint per probe phase, bowl position, robot-bowl contacts."""
import json, sys
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state, drive
world, state_f = sys.argv[1], sys.argv[2]
sc = Scene(world); sc.reset()
st = json.load(open(state_f))["objects"]; apply_state(sc, st)
o = sc.world["objects"][sc.names.index("bowl")]
r = float(o["attrs"].get("radius", 0.05)); top = sc.world["table"]["height"]
bx, by = st["bowl"]["pos"][:2]; z = top + min(0.03, o["top_z"] * 0.6)
print("bowl", np.round(sc.env.object_pos("bowl"), 3), "r", r, "eef0", np.round(sc.env.eef_pos(), 3))
acts = []
for _ in range(8):
    sc.env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
for name, tgt, n in [("above", [bx, by + r + 0.03, z + 0.08], 60), ("descend", [bx, by + r + 0.03, z], 60),
                     ("push", [bx, by + r - 0.045, z], 80)]:
    drive(sc, tgt, 1.0, acts, max_steps=n)
    d = sc.env.sim.data
    con = [(sc.env.sim.model.geom_id2name(d.contact[i].geom1), sc.env.sim.model.geom_id2name(d.contact[i].geom2))
           for i in range(d.ncon)]
    con = [c for c in con if any("bowl" in (x or "") for x in c)]
    print(name, "steps", len(acts), "target", np.round(tgt, 3), "eef", np.round(sc.env.eef_pos(), 3),
          "bowl", np.round(sc.env.object_pos("bowl"), 3), "bowl contacts", con[:4])
