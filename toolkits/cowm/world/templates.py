"""CoWM-PO SDK: parametric category templates (frozen, human-written; plan §3.11 L-SDK).

Every template builds one robosuite-compatible object XML from generic geometry (primitives or
revolved profiles + CoACD collision). No LIBERO asset is read. Object frame: origin at bottom centre,
+z up, front faces -x (towards the robot). Each returns {"xml", "free", "top_z", "joints"}.
"""
import os

import numpy as np

SDK_VERSION = "cowm-sdk-0.1"


def _f(v):
    return " ".join(f"{float(x):.5f}" for x in np.atleast_1d(v))


def _part(t, size, pos, rgba, euler=None, friction="1 0.005 0.0001", density=500.0, solref="0.02 1", condim=4):
    """Primitive part: one collision geom (group 0) + one visual twin (group 1, no contact, no mass)."""
    e = f' euler="{euler}"' if euler else ""
    col = (f'<geom type="{t}" size="{_f(size)}" pos="{_f(pos)}"{e} group="0" rgba="{rgba}" '
           f'friction="{friction}" density="{density:.3f}" solref="{solref}" condim="{condim}"/>')
    vis = (f'<geom type="{t}" size="{_f(size)}" pos="{_f(pos)}"{e} group="1" contype="0" conaffinity="0" '
           f'density="0" rgba="{rgba}"/>')
    return col + vis


def _wrap(name, assets, body, top_z, radius):
    return (f'<mujoco model="{name}"><asset>{assets}</asset><worldbody><body>'
            f'<body name="object">{body}</body>'
            f'<site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>'
            f'<site rgba="0 0 0 0" size="0.005" pos="0 0 {top_z:.5f}" name="top_site"/>'
            f'<site rgba="0 0 0 0" size="0.005" pos="{radius:.5f} {radius:.5f} 0" name="horizontal_radius_site"/>'
            f'</body></worldbody></mujoco>')


def _save(out_dir, name, xml):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def _revolved(name, out_dir, profile, rgba, density, friction, max_hulls=12):
    """Revolve a closed (radius, z) profile; visual mesh + CoACD convex collision pieces."""
    import coacd
    import trimesh

    coacd.set_log_level("error")
    mesh = trimesh.creation.revolve(np.asarray(profile, dtype=float), sections=48)
    mesh.fix_normals()
    os.makedirs(out_dir, exist_ok=True)
    mesh.export(os.path.join(out_dir, f"{name}_vis.obj"))
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=0.05, max_convex_hull=max_hulls)
    assets = [f'<mesh name="{name}_vis" file="{name}_vis.obj"/>']
    geoms = [f'<geom type="mesh" mesh="{name}_vis" group="1" contype="0" conaffinity="0" density="0" rgba="{rgba}"/>']
    for i, (v, fc) in enumerate(parts):
        trimesh.Trimesh(v, fc).export(os.path.join(out_dir, f"{name}_c{i}.obj"))
        assets.append(f'<mesh name="{name}_c{i}" file="{name}_c{i}.obj"/>')
        geoms.append(f'<geom type="mesh" mesh="{name}_c{i}" group="0" rgba="{rgba}" '
                     f'friction="{friction:.4f} 0.005 0.0001" density="{density:.3f}" condim="4"/>')
    return "".join(assets), "".join(geoms), len(parts)


def bowl(name, out_dir, radius=0.07, height=0.05, wall=0.006, density=1500.0, friction=0.9, rgba="0.1 0.1 0.1 1"):
    b = 0.5 * radius
    prof = [(0, 0), (b, 0), (radius, height), (radius - wall, height), (b - 0.5 * wall, wall), (0, wall)]
    a, g, _ = _revolved(name, out_dir, prof, rgba, density, friction)
    return {"xml": _save(out_dir, name, _wrap(name, a, g, height, radius)), "free": True,
            "top_z": height, "joints": []}


def plate(name, out_dir, radius=0.10, height=0.018, density=1500.0, friction=0.9, rgba="0.92 0.92 0.9 1"):
    r0 = 0.6 * radius
    prof = [(0, 0), (r0, 0), (radius, height), (radius - 0.006, height), (r0 - 0.004, 0.004), (0, 0.004)]
    a, g, _ = _revolved(name, out_dir, prof, rgba, density, friction)
    return {"xml": _save(out_dir, name, _wrap(name, a, g, height, radius)), "free": True,
            "top_z": height, "joints": []}


def bottle(name, out_dir, radius=0.035, height=0.28, neck_radius=0.013, density=900.0, friction=0.9,
           rgba="0.15 0.3 0.15 1"):
    h1 = 0.62 * height
    sh = 0.12 * height
    body = (_part("cylinder", [radius, h1 / 2], [0, 0, h1 / 2], rgba, friction=f"{friction:.4f} 0.005 0.0001",
                  density=density)
            + _part("cylinder", [0.5 * (radius + neck_radius), sh / 2], [0, 0, h1 + sh / 2], rgba,
                    friction=f"{friction:.4f} 0.005 0.0001", density=density)
            + _part("cylinder", [neck_radius, (height - h1 - sh) / 2], [0, 0, (height + h1 + sh) / 2], rgba,
                    friction=f"{friction:.4f} 0.005 0.0001", density=density))
    return {"xml": _save(out_dir, name, _wrap(name, "", body, height, radius)), "free": True,
            "top_z": height, "joints": []}


def box(name, out_dir, size_x=0.07, size_y=0.05, size_z=0.025, density=600.0, friction=0.9, rgba="0.9 0.85 0.6 1"):
    body = _part("box", [size_x / 2, size_y / 2, size_z / 2], [0, 0, size_z / 2], rgba,
                 friction=f"{friction:.4f} 0.005 0.0001", density=density)
    return {"xml": _save(out_dir, name, _wrap(name, "", body, size_z, max(size_x, size_y) / 2)), "free": True,
            "top_z": size_z, "joints": []}


def drawer_cabinet(name, out_dir, width=0.30, depth=0.26, height=0.36, n_drawers=3, drawer_damping=5.0,
                   drawer_travel=0.16, rgba="0.55 0.38 0.22 1"):
    """Frame open at the front (-x) with n stacked drawers on slide joints (open = -x)."""
    t, W, D, H, n = 0.01, width, depth, height, int(n_drawers)
    fr = ""
    fr += _part("box", [D / 2, W / 2, t / 2], [0, 0, t / 2], rgba)
    fr += _part("box", [D / 2, W / 2, t / 2], [0, 0, H - t / 2], rgba)
    fr += _part("box", [t / 2, W / 2, H / 2], [D / 2 - t / 2, 0, H / 2], rgba)
    for s in (-1, 1):
        fr += _part("box", [D / 2, t / 2, H / 2], [0, s * (W / 2 - t / 2), H / 2], rgba)
    hd = (H - 2 * t) / n
    for k in range(1, n):
        fr += _part("box", [D / 2 - t, W / 2 - t, t / 4], [0, 0, t + k * hd], rgba)
    labels = {3: ["bottom", "middle", "top"], 2: ["bottom", "top"]}.get(n, [f"d{i}" for i in range(n)])
    dw, dh = W - 2 * t - 0.006, hd - 0.012
    x_front, x_back = -D / 2 + 0.005, D / 2 - t - 0.004
    L = x_back - x_front
    xc = 0.5 * (x_front + x_back)
    joints = []
    for i, lab in enumerate(labels):
        z0 = t + i * hd + 0.003
        drw = (_part("box", [L / 2, dw / 2, 0.003], [xc, 0, 0.003], rgba)
               + _part("box", [0.005, dw / 2 + 0.002, dh / 2], [x_front, 0, dh / 2], rgba)
               + _part("box", [0.003, dw / 2, dh / 2], [x_back, 0, dh / 2], rgba))
        for s in (-1, 1):
            drw += _part("box", [L / 2, 0.003, dh / 2], [xc, s * (dw / 2 - 0.003), dh / 2], rgba)
        drw += _part("box", [0.008, 0.04, 0.006], [x_front - 0.013, 0, dh / 2], "0.7 0.7 0.7 1")
        fr += (f'<body name="drawer_{lab}" pos="0 0 {z0:.5f}">'
               f'<joint name="drawer_{lab}" type="slide" axis="1 0 0" limited="true" '
               f'range="{-drawer_travel:.4f} 0" damping="{drawer_damping:.4f}"/>{drw}</body>')
        joints.append(f"drawer_{lab}")
    return {"xml": _save(out_dir, name, _wrap(name, "", fr, H, max(W, D) / 2)), "free": False,
            "top_z": H, "joints": joints}


def stove(name, out_dir, width=0.24, depth=0.22, height=0.04, knob_damping=0.05, rgba="0.25 0.25 0.27 1"):
    W, D, h = width, depth, height
    body = _part("box", [D / 2, W / 2, h / 2], [0, 0, h / 2], rgba)
    body += (f'<geom type="cylinder" size="0.06 0.002" pos="0.02 0 {h + 0.002:.5f}" group="1" contype="0" '
             f'conaffinity="0" density="0" rgba="0.5 0.1 0.1 1"/>')
    body += (f'<body name="knob" pos="{-D / 2 - 0.010:.5f} {0.3 * W:.5f} {h / 2:.5f}">'
             f'<joint name="knob" type="hinge" axis="1 0 0" limited="true" range="0 1.5708" '
             f'damping="{knob_damping:.4f}"/>'
             + _part("cylinder", [0.012, 0.010], [0, 0, 0], "0.8 0.8 0.8 1", euler="0 1.5708 0", density=300.0)
             + "</body>")
    return {"xml": _save(out_dir, name, _wrap(name, "", body, h, max(W, D) / 2)), "free": False,
            "top_z": h, "joints": ["knob"]}


def rack(name, out_dir, width=0.24, depth=0.16, rgba="0.45 0.3 0.2 1"):
    body = _part("box", [depth / 2, width / 2, 0.005], [0, 0, 0.005], rgba)
    for s in (-1, 1):
        body += _part("box", [0.006, width / 2, 0.02], [s * 0.035, 0, 0.03], rgba)
        body += _part("cylinder", [0.008, width / 2], [s * 0.035, 0, 0.055], rgba, euler="1.5708 0 0")
    return {"xml": _save(out_dir, name, _wrap(name, "", body, 0.063, max(width, depth) / 2)), "free": False,
            "top_z": 0.063, "joints": []}


TEMPLATES = {"bowl": bowl, "plate": plate, "bottle": bottle, "box": box,
             "drawer_cabinet": drawer_cabinet, "stove": stove, "rack": rack}
