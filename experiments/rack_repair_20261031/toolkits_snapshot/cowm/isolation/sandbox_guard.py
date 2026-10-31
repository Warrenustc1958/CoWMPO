"""CoWM-PO Builder sandbox guard, installed as sitecustomize.py in the Builder venv (plan §3.7C).

Defense in depth only. The real boundaries are kernel level (see run_builder.sh): a venv without
libero, a private mount namespace that hides Target paths, a read-only evidence mount and no network.
This hook blocks Target imports and foreign subprocesses, and logs every violation.
"""
import os
import sys

_FORBIDDEN_MODULES = ("libero", "bddl", "rlinf.envs.sim.libero")
_ENABLED = os.environ.get("COWM_SANDBOX") == "1"
_LOG = os.environ.get("COWM_GUARD_LOG", "/tmp/cowm_guard.log")


def _log(msg):
    try:
        with open(_LOG, "a") as f:
            f.write(msg + "\n")
    except OSError:
        pass


class _BlockTargetImports:
    def find_spec(self, name, path=None, target=None):
        if any(name == m or name.startswith(m + ".") for m in _FORBIDDEN_MODULES):
            _log(f"BLOCK import {name}")
            raise ImportError(f"[cowm sandbox] import of Target module '{name}' is forbidden")
        return None


def _audit(event, args):
    if event in ("subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn", "os.system"):
        exe = args[0] if args else None
        if event == "subprocess.Popen" and exe is None and len(args) > 1 and args[1]:
            argv = args[1]
            exe = argv[0] if isinstance(argv, (list, tuple)) else str(argv).split()[0]
        exe = os.fsdecode(exe) if isinstance(exe, (bytes, str)) else str(exe)
        if event == "os.system" or os.path.realpath(exe) != os.path.realpath(sys.executable):
            _log(f"BLOCK {event} {exe}")
            raise PermissionError(f"[cowm sandbox] {event} of '{exe}' is forbidden")


if _ENABLED:
    sys.meta_path.insert(0, _BlockTargetImports())
    sys.addaudithook(_audit)
