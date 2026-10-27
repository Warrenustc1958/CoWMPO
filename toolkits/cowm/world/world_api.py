"""CoWM-PO SDK: the ONLY interface a Code Agent's world_program.py may use (plan §3.11 L-Agent).

A world program is a single function `build(w)` that declares a static world: a table, template
instances, identifiable parameters and library mechanisms. There is deliberately no time hook,
no per-step callback, no state write and no teleport: the world can only evolve through physics
driven by the policy's actions. Execution never calls the Agent.
"""
import math

from .templates import TEMPLATES


class ParamRef:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"ParamRef({self.name!r})"


MECHANISMS = {
    # name: (target kind, MJCF attribute)  -- constant, state-independent edits only
    "joint_damping": ("joint", "damping"),
    "joint_frictionloss": ("joint", "frictionloss"),
    "geom_friction": ("object", "friction"),
}


class WorldAPI:
    def __init__(self):
        self._params = {}
        self._objects = []
        self._table = None
        self._camera = None
        self._mechanisms = []

    # ---- identifiable parameters ------------------------------------------------------------
    def p(self, name, init, bounds):
        """Declare a parameter to be identified later (S6). Returns a reference usable anywhere a number is."""
        lo, hi = float(bounds[0]), float(bounds[1])
        if not (lo <= float(init) <= hi):
            raise ValueError(f"param {name}: init {init} outside bounds {bounds}")
        if name in self._params:
            raise ValueError(f"param {name} declared twice")
        self._params[name] = {"init": float(init), "bounds": [lo, hi]}
        return ParamRef(name)

    # ---- scene ------------------------------------------------------------------------------
    def table(self, size=(1.0, 1.2, 0.05), height=0.8, friction=(1.0, 0.005, 0.0001)):
        self._table = {"size": list(size), "height": height, "friction": list(friction)}

    def add(self, category, name, xy, yaw=0.0, **attrs):
        """Place one template instance. Fixed vs free is decided by the template, not the program."""
        if category not in TEMPLATES:
            raise ValueError(f"unknown category {category!r}; available: {sorted(TEMPLATES)}")
        if any(o["name"] == name for o in self._objects):
            raise ValueError(f"object name {name!r} used twice")
        self._objects.append({"category": category, "name": name, "xy": list(xy), "yaw": yaw, "attrs": attrs})

    def camera(self, pos, quat, fovy=45.0, name="agentview"):
        self._camera = {"name": name, "pos": list(pos), "quat": list(quat), "fovy": fovy}

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
