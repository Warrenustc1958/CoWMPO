"""L1 batch-partition independence test (Step 3, CPU/OSMesa). Exit code 1 on any failure.

Uses the REAL L1 backend (CodeBackend._scene_for / _open_l1): one Scene per env slot, several Scenes in one
process (also exercises GL-context ownership). Each episode = open (reset + warmup + S5 state) then a fixed
action sequence; per step record agentview RGB, qpos, qvel.
  partition 1 : slot 0 runs [e0, e1, e2, e3] in order
  partition 2 : slot 0 runs [e2, e0], slot 1 runs [e3, e1], interleaved step by step
  partition 3 : a fresh backend, slots 0..3 one episode each, interleaved
Every episode's record must be bitwise identical across the three partitions.
usage: python test_batch_partition.py <repo_root> <code.py> <world> <state_root> <ep0,ep1,ep2,ep3>
"""
import importlib.util
import os
import sys
import zlib

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

N_STEPS = 16


def actions(ep):
    rng = np.random.default_rng(zlib.crc32(ep.encode()))   # per-episode, partition- and process-independent
    return [np.concatenate([rng.uniform(-1, 1, 6) * 0.4, [rng.choice([-1.0, 1.0])]]) for _ in range(N_STEPS)]


def backend(repo, code_py, world, state_root):
    sys.path.insert(1, repo)
    import rlinf.envs.sim.world_model.backend as pkg  # noqa: F401
    name = "rlinf.envs.sim.world_model.backend.code"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, code_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    be = object.__new__(sys.modules[name].CodeBackend)
    be.world_dir, be.state_dir, be.world_friction, be._envs, be._first = world, state_root, {}, {}, {}
    return be


def run(be, schedule):
    """schedule: {slot: [episodes]}; episodes of different slots advance interleaved, one step each."""
    out, queues = {}, {s: list(eps) for s, eps in schedule.items()}
    active = {}
    while queues or active:
        for s in list(queues):
            if s not in active and queues[s]:
                ep = queues[s].pop(0)
                be._open_l1(s, ep)
                first = (be._first[s].numpy() * 1e6).round().astype(np.int64).ravel()
                active[s] = (ep, actions(ep), [first])
            if not queues[s]:
                del queues[s]
        for s in sorted(active):
            ep, acts, rec = active[s]
            env = be._envs[s]["env"]
            obs, _, _, _ = env.step(acts[len(rec) - 1])
            d = env.sim.data
            rec.append(np.concatenate([obs["agentview_image"].astype(np.float64).ravel(), d.qpos, d.qvel]))
            if len(rec) == N_STEPS + 1:
                out[ep] = rec
                del active[s]
    return out


def close(be):
    for v in be._envs.values():
        v["env"].close()


def main():
    repo, code_py, world, state_root, eps = sys.argv[1:6]
    e = eps.split(",")
    b1 = backend(repo, code_py, world, state_root)
    p1 = run(b1, {0: [e[0], e[1], e[2], e[3]]})
    p2 = run(b1, {0: [e[2], e[0]], 1: [e[3], e[1]]})          # same backend, reused + new slot
    close(b1)
    b3 = backend(repo, code_py, world, state_root)
    p3 = run(b3, {0: [e[0]], 1: [e[1]], 2: [e[2]], 3: [e[3]]})
    close(b3)
    same = lambda a, b: len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
    res = {ep: {"p1_vs_p2": same(p1[ep], p2[ep]), "p1_vs_p3": same(p1[ep], p3[ep])} for ep in e}
    ok = all(all(v.values()) for v in res.values())
    print(res)
    print("BATCH_PARTITION_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
