"""Runtime overrides of identifiable dynamics parameters on a compiled world (no recompilation).

  <material>.friction  : sliding friction of every geom of objects with that material; for the table
                         material also the arena table collision geom
  <material>.density   : mass/inertia of objects with that material scaled by value / compiled value
Shape parameters change geometry and need recompilation (compile_world); they are not handled here.
MuJoCo combines two contacting geoms' friction with max(), so an object's own friction only matters
when it exceeds the table's -- this is why the table friction is the one that controls sliding.
"""
import numpy as np


def _baseline(scene):
    if not hasattr(scene, "_param_base"):
        m = scene.env.sim.model
        scene._param_base = {"friction": np.array(m.geom_friction).copy(),
                             "mass": np.array(m.body_mass).copy(), "inertia": np.array(m.body_inertia).copy()}
    return scene._param_base


def apply_params(scene, values):
    """values: {param_name: value}; unknown names raise. Always applied relative to the compiled baseline."""
    m = scene.env.sim.model
    base = _baseline(scene)
    m.geom_friction[:] = base["friction"]
    m.body_mass[:] = base["mass"]
    m.body_inertia[:] = base["inertia"]
    world = scene.world
    table_mat = world["table"].get("material")
    for name, v in (values or {}).items():
        mat, kind = name.rsplit(".", 1)
        if kind == "friction":
            hit = 0
            for k, o in enumerate(world["objects"]):
                if o.get("material") == mat:
                    m.geom_friction[scene.geom2obj == k, 0] = v
                    hit += 1
            if mat == table_mat:
                for g in range(m.ngeom):
                    if (m.geom_id2name(g) or "").startswith("table_collision"):
                        m.geom_friction[g, 0] = v
                        hit += 1
            if not hit:
                raise ValueError(f"{name}: no geom uses material {mat}")
        elif kind == "density":
            d0 = world["params"][name]["value"]
            scale = float(v) / d0
            for k, o in enumerate(world["objects"]):
                if o.get("material") == mat:
                    root = m.body_name2id(scene.env.objects[k].root_body)
                    sel = np.array([int(m.body_rootid[b]) == root for b in range(m.nbody)])
                    m.body_mass[sel] = base["mass"][sel] * scale
                    m.body_inertia[sel] = base["inertia"][sel] * scale
        else:
            raise ValueError(f"unsupported runtime parameter {name}")
    scene.env.sim.forward()
