"""Runtime overrides of identifiable dynamics parameters on a compiled world (no recompilation).

  <material>.friction  : sliding friction of every geom of objects with that material; for the table
                         material also the arena table collision geom
  <material>.density   : mass/inertia of objects with that material scaled by value / compiled value
Shape parameters change geometry and need recompilation (compile_world); they are not handled here.
Contact rule "legacy" (default): MuJoCo combines two contacting geoms' friction with max(), so an object's
own friction only matters when it exceeds the table's. Contact rule "object_table_pair" (opt-in, SDK 0.2.2):
object-table contacts are explicit <pair>s and <material>.friction also sets their pair friction, so the
object's material controls its sliding on the table; the table material then only affects the remaining
(robot/fixture-table) contacts. Note <material>.friction still also changes that material's gripper and
object-object contacts (one shared material value; audit 10.28).
"""
import numpy as np


def _baseline(scene):
    if not hasattr(scene, "_param_base"):
        m = scene.env.sim.model
        scene._param_base = {"friction": np.array(m.geom_friction).copy(),
                             "pair_friction": np.array(m.pair_friction).copy(),
                             "mass": np.array(m.body_mass).copy(), "inertia": np.array(m.body_inertia).copy()}
    return scene._param_base


def apply_params(scene, values):
    """Apply validated overrides relative to the compiled baseline; invalid input leaves the model intact."""
    m = scene.env.sim.model
    base = _baseline(scene)
    friction, pairs = base["friction"].copy(), base["pair_friction"].copy()
    mass, inertia = base["mass"].copy(), base["inertia"].copy()
    world = scene.world
    table_mat = world["table"].get("material")
    for name, v in (values or {}).items():
        if not isinstance(name, str) or "." not in name:
            raise ValueError(f"unsupported runtime parameter {name!r}")
        mat, kind = name.rsplit(".", 1)
        v = float(v)
        if not np.isfinite(v) or v < 0 or kind == "density" and v == 0:
            raise ValueError(f"{name}: expected finite {'positive' if kind == 'density' else 'nonnegative'} value")
        spec = world.get("params", {}).get(name, {})
        if "bounds" in spec and not spec["bounds"][0] <= v <= spec["bounds"][1]:
            raise ValueError(f"{name}: {v} outside declared bounds {spec['bounds']}")
        if kind == "friction":
            hit = 0
            for k, o in enumerate(world["objects"]):
                if o.get("material") == mat:
                    sel = scene.geom2obj == k
                    friction[sel, 0] = v
                    hit += int(sel.sum())
            for p in range(m.npair):   # object-table <pair>s (contact rule object_table_pair)
                for g in (int(m.pair_geom1[p]), int(m.pair_geom2[p])):
                    k = int(scene.geom2obj[g])
                    if k >= 0 and world["objects"][k].get("material") == mat:
                        pairs[p, 0:2] = v
            if mat == table_mat:
                for g in range(m.ngeom):
                    if (m.geom_id2name(g) or "").startswith("table_collision"):
                        friction[g, 0] = v
                        hit += 1
            if not hit:
                raise ValueError(f"{name}: no geom uses material {mat}")
        elif kind == "density":
            d0 = spec.get("value")
            if d0 is None or not np.isfinite(d0) or d0 <= 0:
                raise ValueError(f"{name}: missing positive compiled density")
            scale = float(v) / d0
            hit = False
            for k, o in enumerate(world["objects"]):
                if o.get("material") == mat:
                    root = m.body_name2id(scene.env.objects[k].root_body)
                    sel = np.array([int(m.body_rootid[b]) == root for b in range(m.nbody)])
                    mass[sel] = base["mass"][sel] * scale
                    inertia[sel] = base["inertia"][sel] * scale
                    hit = hit or bool(sel.any())
            if not hit:
                raise ValueError(f"{name}: no body uses material {mat}")
        else:
            raise ValueError(f"unsupported runtime parameter {name}")
    if not all(np.isfinite(a).all() for a in (friction, pairs, mass, inertia)):
        raise ValueError("runtime overrides produced nonfinite model parameters")
    m.geom_friction[:] = friction
    m.pair_friction[:] = pairs
    m.body_mass[:] = mass
    m.body_inertia[:] = inertia
    scene.env.sim.forward()
