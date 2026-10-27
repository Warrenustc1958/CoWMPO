"""Apply the two small CoWM-PO hooks to RLinf (idempotent)."""
import sys

repo = sys.argv[1]

env_py = f"{repo}/rlinf/envs/sim/world_model/env.py"
src = open(env_py).read()
anchor = ("        # One seed is shared by the batch; per-trajectory seeds are future work.\n"
          "        self.backend.close_session(target_slots)\n")
hook = ("        # Optional hook: tell the backend which episode file each slot starts (Code WM needs it).\n"
        "        set_episodes = getattr(self.backend, \"set_session_episodes\", None)\n"
        "        if set_episodes is not None:\n"
        "            set_episodes(\n"
        "                target_slots,\n"
        "                [self.dataset.npy_files[int(i)] for i in episode_indices],\n"
        "            )\n")
if "set_session_episodes" not in src:
    assert src.count(anchor) == 1, "env.py anchor not found exactly once"
    src = src.replace(anchor, hook + anchor)
    open(env_py, "w").write(src)
    print("patched", env_py)
else:
    print("already patched", env_py)

reg_py = f"{repo}/rlinf/envs/sim/world_model/registry.py"
src = open(reg_py).read()
line = '    "opensora": "rlinf.envs.sim.world_model.backend.opensora",\n'
new = line + '    "code": "rlinf.envs.sim.world_model.backend.code",\n'
if '"code":' not in src:
    assert src.count(line) == 1, "registry anchor not found exactly once"
    open(reg_py, "w").write(src.replace(line, new))
    print("patched", reg_py)
else:
    print("already patched", reg_py)
