"""S1 gate: run INSIDE run_builder.sh. Every 'blocked' check must fail, every 'allowed' check must work."""
import json
import os
import socket
import sys

EVIDENCE = os.environ.get("COWM_EVIDENCE", "/data_cf/cowmpo/evidence")
WORLDS = os.environ.get("COWM_WORLDS", "/data_cf/cowmpo/worlds")
TARGET_BOWL = ("/data_cf/cowmpo/.venv-oft-wan/lib/python3.11/site-packages/libero/libero/assets/"
               "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml")
LEAK_NAMES = ("akita_black_bowl", "stable_scanned_objects", "bddl_files", "init_files", ".pruned_init",
              ".bddl", "rlinf_libero", "rlinf-libero")


def _import(name):
    __import__(name)


def _walk_for_leaks():
    hits = []
    for root in ("/mnt/nfs", "/data_cf", "/root", "/home", "/tmp", "/opt"):
        for dirpath, dirnames, filenames in os.walk(root):
            for n in dirnames + filenames:
                if any(k in n for k in LEAK_NAMES):
                    hits.append(os.path.join(dirpath, n))
            # a python package called libero anywhere visible
            if os.path.basename(dirpath) == "libero" and "__init__.py" in filenames:
                hits.append(dirpath)
    if hits:
        raise AssertionError(f"visible Target artifacts: {hits[:5]}")


def _sys_path_has_libero():
    for p in sys.path:
        if p and os.path.isdir(os.path.join(p, "libero")):
            raise AssertionError(f"libero package on sys.path: {p}")


def _net():
    socket.create_connection(("1.1.1.1", 443), timeout=3)


def _dns():
    socket.getaddrinfo("huggingface.co", 443)


def _write_evidence():
    with open(os.path.join(EVIDENCE, "_should_fail.txt"), "w") as f:
        f.write("x")


def _subprocess():
    import subprocess
    subprocess.run(["cat", "/etc/hostname"], check=True, capture_output=True)


def _read_evidence():
    with open(os.path.join(EVIDENCE, "_selftest.json")) as f:
        assert json.load(f)["ok"] is True


def _write_worlds():
    p = os.path.join(WORLDS, "_selftest_write.txt")
    with open(p, "w") as f:
        f.write("ok")
    os.remove(p)


def _executor_runs():
    import numpy as np
    import robosuite as suite
    from robosuite.environments.manipulation.lift import Lift
    env = Lift(robots="Panda", controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
               use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False, control_freq=20,
               initialization_noise=None)
    env.reset()
    for _ in range(5):
        env.step(np.zeros(7))


def _coacd():
    import coacd  # noqa: F401
    import trimesh
    m = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    assert m.is_watertight


BLOCKED = [
    ("import libero", lambda: _import("libero")),
    ("import bddl", lambda: _import("bddl")),
    ("read Target bowl xml", lambda: open(TARGET_BOWL).read()),
    ("list Target venv", lambda: os.listdir("/data_cf/cowmpo/.venv-oft-wan/lib")),
    ("list Wan dataset", lambda: os.listdir("/mnt/nfs/fjj/Models/RLinf-Wan-LIBERO-Goal/dataset")),
    ("list CWS data", lambda: os.listdir("/data_cf/cws")),
    ("network tcp", _net),
    ("network dns", _dns),
    ("write evidence", _write_evidence),
    ("foreign subprocess", _subprocess),
]
MUST_HOLD = [
    ("no Target artifacts visible", _walk_for_leaks),
    ("no libero on sys.path", _sys_path_has_libero),
    ("read evidence", _read_evidence),
    ("write worlds", _write_worlds),
    ("robosuite Panda executor runs", _executor_runs),
    ("coacd + trimesh import", _coacd),
]


def main():
    ok = True
    print(f"python={sys.executable} sandbox={os.environ.get('COWM_SANDBOX')}")
    for name, fn in BLOCKED:
        try:
            fn()
            res, good = "NOT BLOCKED", False
        except Exception as e:  # noqa: BLE001
            res, good = f"blocked ({type(e).__name__})", True
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] blocked : {name:32s} {res}")
    for name, fn in MUST_HOLD:
        try:
            fn()
            res, good = "ok", True
        except Exception as e:  # noqa: BLE001
            res, good = f"{type(e).__name__}: {str(e)[:100]}", False
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] holds   : {name:32s} {res}")
    print("S1_ISOLATION", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
