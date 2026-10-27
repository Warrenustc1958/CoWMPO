"""CoWM-PO SDK: compile an Agent world program into an executable, versioned world directory.

  world_program.py --lint--> restricted exec of build(w) --> templates --> world.json + assets/
Runs inside the Builder sandbox (S1). The output is frozen; execution never calls the Agent.
Materials become shared identifiable parameters "<material>.density" / "<material>.friction" with bounds
from priors.MATERIALS. Camera and table height come from the declared platform prior.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
import xml.etree.ElementTree as ET

from .lint import SAFE_CALLS, lint
from .priors import MATERIALS, platform_prior
from .templates import SDK_VERSION, TEMPLATES
from .world_api import MECHANISMS, MaterialRef, WorldAPI, resolve, yaw_quat

_SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                  for k in SAFE_CALLS}


def _apply_mechanism(xml_path, obj_name, mech):
    tree = ET.parse(xml_path)
    attr = MECHANISMS[mech["kind"]][1]
    hit = 0
    for j in tree.iter("joint"):
        if j.get("name") == mech["joint"]:
            j.set(attr, f"{float(mech['value']):.5f}")
            hit += 1
    if hit == 0:
        raise ValueError(f"mechanism {mech['kind']} on {obj_name}: joint {mech['joint']!r} not found")
    tree.write(xml_path)


def compile_world(program_path, out_dir, param_values=None):
    src = open(program_path).read()
    lint(src)
    w = WorldAPI()
    ns = {"__builtins__": _SAFE_BUILTINS, "math": math}
    exec(compile(src, program_path, "exec"), ns)
    ns["build"](w)

    # material classes -> shared parameters
    params = dict(w._params)
    for mname, cls in w._materials.items():
        for key in ("density", "friction"):
            init, lo, hi = MATERIALS[cls][key]
            params[f"{mname}.{key}"] = {"init": init, "bounds": [lo, hi], "material_class": cls}
    values = {}
    for name, spec in params.items():
        v = float((param_values or {}).get(name, spec["init"]))
        lo, hi = spec["bounds"]
        values[name] = min(max(v, lo), hi)

    plat = platform_prior()
    t = w._table or {"size": [1.0, 1.2, 0.05], "material": None}
    tmat = t["material"]
    table = {"size": [float(v) for v in resolve(t["size"], values)], "height": plat["table_height"],
             "friction": [values[f"{tmat.name}.friction"] if isinstance(tmat, MaterialRef) else 1.0, 0.005, 0.0001],
             "material": tmat.name if isinstance(tmat, MaterialRef) else None}
    top = table["height"]

    objects = []
    for o in w._objects:
        attrs = {k: resolve(v, values) for k, v in o["attrs"].items()}
        fn = TEMPLATES[o["category"]]
        sig = inspect.signature(fn).parameters
        m = o["material"]
        if isinstance(m, MaterialRef):
            for key in ("density", "friction"):
                if key in sig:
                    attrs[key] = values[f"{m.name}.{key}"]
        res = fn(o["name"], os.path.join(out_dir, "assets", o["name"]), **attrs)
        x, y = o["xy"]
        lift = 0.003 if res["free"] else 0.0
        objects.append({"name": o["name"], "category": o["category"], "xml": os.path.relpath(res["xml"], out_dir),
                        "free": res["free"], "pos": [x, y, top + lift], "quat": yaw_quat(o["yaw"]),
                        "yaw": o["yaw"], "top_z": res["top_z"], "joints": res["joints"],
                        "material": m.name if isinstance(m, MaterialRef) else None, "attrs": attrs})
    for mech in w._mechanisms:
        tgt = next((o for o in objects if o["name"] == mech["obj"]), None)
        if tgt is None:
            raise ValueError(f"mechanism target object {mech['obj']!r} not found")
        _apply_mechanism(os.path.join(out_dir, tgt["xml"]), mech["obj"],
                         dict(mech, value=resolve(mech["value"], values)))

    world = {"sdk_version": SDK_VERSION,
             "program_sha256": hashlib.sha256(src.encode()).hexdigest(),
             "table": table, "camera": plat["camera"], "platform_prior_source": plat["source"],
             "objects": objects, "materials": dict(w._materials),
             "mechanisms": [dict(mm, value=resolve(mm["value"], values)) for mm in w._mechanisms],
             "params": {k: dict(v, value=values[k]) for k, v in params.items()},
             "state_note": "object pos/quat are nominal per-episode state, re-estimated per episode (S5)"}
    blob = json.dumps(world, sort_keys=True).encode()
    world["world_sha256"] = hashlib.sha256(blob).hexdigest()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "world.json"), "w") as f:
        json.dump(world, f, indent=2)
    with open(os.path.join(out_dir, "world_program.py"), "w") as f:
        f.write(src)
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--program", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--params", default=None, help="json {name: value}; defaults to declared init")
    a = p.parse_args()
    vals = json.load(open(a.params)) if a.params else None
    print("compiled", compile_world(a.program, a.out, vals))


if __name__ == "__main__":
    main()
