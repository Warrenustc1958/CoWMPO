"""CoWM-PO SDK: compile an Agent world program into an executable, versioned world directory.

  world_program.py --lint--> restricted exec of build(w) --> templates --> world.json + assets/
Runs inside the Builder sandbox (S1). The output is frozen; execution never calls the Agent.
"""
import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET

from .lint import SAFE_CALLS, lint
from .templates import SDK_VERSION, TEMPLATES
from .world_api import MECHANISMS, WorldAPI, resolve, yaw_quat

_SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                  for k in SAFE_CALLS}


def _apply_mechanism(xml_path, obj_name, mech):
    tree = ET.parse(xml_path)
    kind, attr = mech["kind"], MECHANISMS[mech["kind"]][1]
    val = mech["value"]
    hit = 0
    if MECHANISMS[kind][0] == "joint":
        for j in tree.iter("joint"):
            if j.get("name") == mech["joint"]:
                j.set(attr, f"{float(val):.5f}")
                hit += 1
    else:
        for g in tree.iter("geom"):
            if g.get("group") == "0":
                g.set(attr, f"{float(val):.4f} 0.005 0.0001")
                hit += 1
    if hit == 0:
        raise ValueError(f"mechanism {kind} on {obj_name}: no target found")
    tree.write(xml_path)


def compile_world(program_path, out_dir, param_values=None):
    src = open(program_path).read()
    lint(src)
    w = WorldAPI()
    ns = {"__builtins__": _SAFE_BUILTINS, "math": math}
    exec(compile(src, program_path, "exec"), ns)
    ns["build"](w)

    values = {}
    for name, spec in w._params.items():
        v = float((param_values or {}).get(name, spec["init"]))
        lo, hi = spec["bounds"]
        values[name] = min(max(v, lo), hi)

    table = w._table or {"size": [1.0, 1.2, 0.05], "height": 0.8, "friction": [1.0, 0.005, 0.0001]}
    table = resolve(table, values)
    top = float(table["height"])
    objects = []
    for o in w._objects:
        attrs = {k: resolve(v, values) for k, v in o["attrs"].items()}
        res = TEMPLATES[o["category"]](o["name"], os.path.join(out_dir, "assets", o["name"]), **attrs)
        x, y = (float(v) for v in resolve(o["xy"], values))
        lift = 0.003 if res["free"] else 0.0
        objects.append({"name": o["name"], "category": o["category"], "xml": os.path.relpath(res["xml"], out_dir),
                        "free": res["free"], "pos": [x, y, top + lift],
                        "quat": yaw_quat(float(resolve(o["yaw"], values))), "joints": res["joints"],
                        "attrs": attrs})
    for m in w._mechanisms:
        tgt = next((o for o in objects if o["name"] == m["obj"]), None)
        if tgt is None:
            raise ValueError(f"mechanism target object {m['obj']!r} not found")
        _apply_mechanism(os.path.join(out_dir, tgt["xml"]), m["obj"], dict(m, value=resolve(m["value"], values)))

    world = {"sdk_version": SDK_VERSION,
             "program_sha256": hashlib.sha256(src.encode()).hexdigest(),
             "table": table, "camera": resolve(w._camera, values) if w._camera else None,
             "objects": objects, "mechanisms": [dict(m, value=resolve(m["value"], values)) for m in w._mechanisms],
             "params": {k: dict(v, value=values[k]) for k, v in w._params.items()}}
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
