"""Is a rebuilt world dynamically equivalent to the registered one? (CoACD meshes differ by hash.)

For each (old, new) world pair:
  hulls    : per object, number of collision meshes, sorted hull volumes, union AABB (trimesh)
  compiled : per object body: MuJoCo body mass and inertia after compilation
  replay   : same fixed action sequence (60 steps, pushes through the table area) from reset in both
             worlds -> max |qpos| difference of every free object over the run, and final object xy diff
Equivalent here means: same hull count, total volume within 1%, AABB within 1 mm, mass/inertia within 1%,
replay object positions within 1 mm. Anything else is reported as a different collision geometry.
usage: python coacd_equiv_check.py <registry.json> <old>:<new> [<old>:<new> ...]
"""
import glob
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import trimesh

from cowm.world.env import CodeWorldEnv


def hulls(wdir):
    out = {}
    for d in sorted(glob.glob(os.path.join(wdir, "assets", "*"))):
        fs = sorted(glob.glob(os.path.join(d, "*_c*.obj")))
        if not fs:
            continue
        ms = [trimesh.load(f, force="mesh") for f in fs]
        vol = sorted(round(float(m.volume) * 1e6, 3) for m in ms)          # cm^3
        lo = np.min([m.bounds[0] for m in ms], 0)
        hi = np.max([m.bounds[1] for m in ms], 0)
        out[os.path.basename(d)] = {"n": len(ms), "vol_cm3": vol, "aabb": [lo.tolist(), hi.tolist()]}
    return out


def compiled_and_replay(wdir, acts):
    env = CodeWorldEnv(wdir, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    m = env.sim.model
    mass = {}
    for ob in env.objects:
        b = m.body_name2id(ob.root_body)
        sel = [k for k in range(m.nbody) if int(m.body_rootid[k]) == b]
        mass[ob.name] = {"mass": float(sum(m.body_mass[k] for k in sel)),
                         "inertia": np.sum([m.body_inertia[k] for k in sel], 0).tolist()}
    env.reset()
    traj = {ob.name: [] for ob, o in zip(env.objects, env.world["objects"]) if o["free"]}
    for a in acts:
        env.step(a)
        for n in traj:
            traj[n].append(env.object_pos(n).copy())
    env.close()
    return mass, {n: np.array(v) for n, v in traj.items()}


def main():
    reg = {v["version"]: v for v in json.load(open(sys.argv[1]))["versions"]}
    rng = np.random.default_rng(0)
    acts = [np.concatenate([rng.uniform(-1, 1, 3) * 0.8, [0, 0, 0], [rng.choice([-1.0, 1.0])]]) for _ in range(60)]
    rows = []
    for pair in sys.argv[2:]:
        o, n = pair.split(":")
        wo, wn = reg[o]["world_dir"], reg[n]["world_dir"]
        ho, hn = hulls(wo), hulls(wn)
        mo, to = compiled_and_replay(wo, acts)
        mn, tn = compiled_and_replay(wn, acts)
        r = {"old": o, "new": n, "objects": {}}
        eq = True
        for obj in sorted(set(ho) | set(hn)):
            a, b = ho.get(obj), hn.get(obj)
            if a is None or b is None:
                r["objects"][obj] = "mesh object missing in one world"
                eq = False
                continue
            dv = abs(sum(a["vol_cm3"]) - sum(b["vol_cm3"])) / max(sum(a["vol_cm3"]), 1e-9)
            daabb = float(np.abs(np.array(a["aabb"]) - np.array(b["aabb"])).max() * 1000)
            dm = abs(mo[obj]["mass"] - mn[obj]["mass"]) / mo[obj]["mass"]
            di = float(np.max(np.abs(np.array(mo[obj]["inertia"]) - np.array(mn[obj]["inertia"])) /
                              np.maximum(np.abs(mo[obj]["inertia"]), 1e-12)))
            dq = float(np.abs(to[obj] - tn[obj]).max() * 1000) if obj in to else None
            same = (a["n"] == b["n"] and dv < 0.01 and daabb < 1.0 and dm < 0.01 and di < 0.01
                    and (dq is None or dq < 1.0))
            eq &= same
            r["objects"][obj] = {"hulls": [a["n"], b["n"]], "vol_rel_diff": round(dv, 4), "aabb_mm": round(daabb, 3),
                                 "mass_rel_diff": round(dm, 4), "inertia_rel_diff": round(di, 4),
                                 "replay_max_mm": None if dq is None else round(dq, 3), "equivalent": same}
        others = {k: float(np.abs(to[k] - tn[k]).max() * 1000) for k in to if k not in r["objects"]}
        r["replay_max_mm_non_mesh"] = {k: round(v, 3) for k, v in others.items()}
        eq &= all(v < 1.0 for v in others.values())
        r["equivalent"] = bool(eq)
        rows.append(r)
        print(json.dumps(r))
    print("COACD_EQUIV", "ALL_EQUIVALENT" if all(r["equivalent"] for r in rows) else "DIFFERENT_GEOMETRY",
          f"{sum(r['equivalent'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
