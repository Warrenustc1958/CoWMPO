"""CoWM-PO SDK: the ONLY interface a Code Agent's world_program.py may use (plan §3.11 L-Agent).

A world program is a single function `build(w)` that declares a static world. There is deliberately no time
hook, no per-step callback, no state write and no teleport: the world can only evolve through physics driven
by the policy's actions. Execution never calls the Agent.

v0.2 rules (SDK 0.2):
  * physical parameters come only from material classes (w.material); no raw density/friction
  * object placements are NOMINAL per-episode state (plain numbers); they are re-estimated per episode by
    the state estimator (S5), so they are never identifiable parameters
  * camera and table height are declared platform priors; the Agent cannot set them
  * every declared parameter must have bounds at least MIN_REL_BOUND_WIDTH * |init| wide
"""
import math

from .priors import MATERIALS, MIN_REL_BOUND_WIDTH
from .templates import TEMPLATES


class ParamRef:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"ParamRef({self.name!r})"


class MaterialRef:
    __slots__ = ("name", "cls")

    def __init__(self, name, cls):
        self.name, self.cls = name, cls


MECHANISMS = {
    # name: (target kind, MJCF attribute)  -- constant, state-independent edits only
    "joint_damping": ("joint", "damping"),
    "joint_frictionloss": ("joint", "frictionloss"),
}


class WorldAPI:
    def __init__(self):
        self._params = {}
        self._materials = {}
        self._objects = []
        self._table = None
        self._mechanisms = []

    # ---- identifiable parameters ------------------------------------------------------------
    def p(self, name, init, bounds):
        """Declare a geometric/mechanism parameter to be identified later (S6). Not for positions."""
        lo, hi = float(bounds[0]), float(bounds[1])
        if not (lo <= float(init) <= hi):
            raise ValueError(f"param {name}: init {init} outside bounds {bounds}")
        if (hi - lo) < MIN_REL_BOUND_WIDTH * abs(float(init)) - 1e-12:
            raise ValueError(f"param {name}: bounds {bounds} narrower than {MIN_REL_BOUND_WIDTH:.0%} of init; "
                             f"do not claim certainty the evidence cannot support")
        if name in self._params:
            raise ValueError(f"param {name} declared twice")
        self._params[name] = {"init": float(init), "bounds": [lo, hi]}
        return ParamRef(name)

    def material(self, name, cls):
        """Declare a material handle of a prior class; objects sharing a handle share density and friction."""
        if cls not in MATERIALS:
            raise ValueError(f"unknown material class {cls!r}; available: {sorted(MATERIALS)}")
        if name in self._materials:
            raise ValueError(f"material {name!r} declared twice")
        self._materials[name] = cls
        return MaterialRef(name, cls)

    # ---- scene ------------------------------------------------------------------------------
    def table(self, size=(1.0, 1.2, 0.05), material=None, **kw):
        if "height" in kw or "friction" in kw:
            raise ValueError("table height is a platform prior and friction comes from material=; "
                             "only size and material may be set")
        if kw:
            raise ValueError(f"unknown table arguments: {sorted(kw)}")
        self._table = {"size": list(size), "material": material}

    def add(self, category, name, xy, yaw=0.0, material=None, **attrs):
        """Place one template instance. xy/yaw are NOMINAL per-episode placements (plain numbers)."""
        if category not in TEMPLATES:
            raise ValueError(f"unknown category {category!r}; available: {sorted(TEMPLATES)}")
        if any(o["name"] == name for o in self._objects):
            raise ValueError(f"object name {name!r} used twice")
        if any(isinstance(v, ParamRef) for v in list(xy) + [yaw]):
            raise ValueError(f"{name}: xy/yaw must be plain numbers. Placements are per-episode state that "
                             f"the state estimator re-measures for every episode; they are not parameters")
        bad = {"density", "friction"} & set(attrs)
        if bad:
            raise ValueError(f"{name}: {sorted(bad)} come from material=w.material(...), not raw values")
        if material is not None and not isinstance(material, MaterialRef):
            raise ValueError(f"{name}: material must be a handle returned by w.material(...)")
        self._objects.append({"category": category, "name": name, "xy": [float(v) for v in xy],
                              "yaw": float(yaw), "material": material, "attrs": attrs})

    def camera(self, *a, **k):
        raise ValueError("the camera is a declared platform prior; world programs cannot set it")

    def mechanism(self, kind, obj, value, joint=None):
        if kind not in MECHANISMS:
            raise ValueError(f"unknown mechanism {kind!r}; available: {sorted(MECHANISMS)}")
        if MECHANISMS[kind][0] == "joint" and joint is None:
            raise ValueError(f"mechanism {kind} needs joint=")
        self._mechanisms.append({"kind": kind, "obj": obj, "joint": joint, "value": value})


def resolve(v, values):
    if isinstance(v, ParamRef):
        return values[v.name]
    if isinstance(v, dict):
        return {k: resolve(x, values) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(resolve(x, values) for x in v)
    return v


def yaw_quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
