"""Step 2 end-to-end parameter-chain test (CPU, simulation only). Exit code 1 on any failure.

  cli_rule        : compile_world CLI --contact-rule object_table_pair writes the rule; contact_rule_of reads it
  legacy_default  : compile without the flag -> legacy (no contact_rule key)
  l1_pair_applied : the REAL L1 backend _scene_for() with world_friction {ceramic: 0.4} on a pair world ->
                    bowl geoms 0.4 AND every bowl-table pair 0.4 (was: pair stayed at the compiled 0.7)
  l1_log          : backend records the applied contact values (applied_contact)
  state_mismatch  : a state estimated for another world is refused by _open_l1
usage: python test_param_chain.py <repo_root> <staged_code.py> <out_dir>
"""
import importlib.util
import json
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

PROG = "/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/world/ref_goal_program_v03.py"
# compiling needs coacd (Builder venv only); the L1 backend needs torch (policy venv): compile in a subprocess
BUILDER_PY = os.environ.get("COWM_BUILDER_PY", "/data_cf/cowmpo/.venv-builder/bin/python")


def main():
    repo, code_py, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    res = {}
    from cowm.world.compile_world import contact_rule_of

    wp, wl = os.path.join(out, "w_pair"), os.path.join(out, "w_legacy")
    subprocess.run([BUILDER_PY, "-m", "cowm.world.compile_world", "--program", PROG, "--out", wp,
                    "--contact-rule", "object_table_pair"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([BUILDER_PY, "-m", "cowm.world.compile_world", "--program", PROG, "--out", wl],
                   check=True, stdout=subprocess.DEVNULL)
    res["cli_rule"] = contact_rule_of(wp) == "object_table_pair"
    res["legacy_default"] = contact_rule_of(wl) == "legacy" and \
        "contact_rule" not in json.load(open(os.path.join(wl, "world.json")))["table"]

    # load the staged backend module under its package name (parent packages come from the repo)
    sys.path.insert(1, repo)
    import rlinf.envs.sim.world_model.backend as pkg  # noqa: F401
    name = "rlinf.envs.sim.world_model.backend.code"
    spec = importlib.util.spec_from_file_location(name, code_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    be = object.__new__(mod.CodeBackend)
    be.world_dir, be.world_friction, be._envs = wp, {"ceramic": 0.4}, {}
    scene = be._scene_for(0)
    m = scene.env.sim.model
    k = scene.names.index("bowl")
    bowl_geoms = [g for g in range(m.ngeom) if int(scene.geom2obj[g]) == k and m.geom_contype[g]]
    bowl_pairs = [p for p in range(m.npair) if k in (int(scene.geom2obj[int(m.pair_geom1[p])]),
                                                        int(scene.geom2obj[int(m.pair_geom2[p])]))]
    res["l1_pair_applied"] = bool(bowl_pairs) and all(abs(m.pair_friction[p, 0] - 0.4) < 1e-9 for p in bowl_pairs) \
        and all(abs(m.geom_friction[g, 0] - 0.4) < 1e-9 for g in bowl_geoms)
    ac = getattr(be, "applied_contact", {})
    res["l1_log"] = ac.get("contact_rule") == "object_table_pair" and 0.4 in ac.get("pair_friction", [])
    sdir = os.path.join(out, "state_other")
    os.makedirs(os.path.join(sdir, "ep0"), exist_ok=True)
    json.dump({"world_sha256": "some-other-world", "objects": {}}, open(os.path.join(sdir, "ep0", "state.json"), "w"))
    be.state_dir = sdir
    try:
        be._open_l1(0, "ep0")
        res["state_mismatch"] = False
    except ValueError:
        res["state_mismatch"] = True
    scene.env.close()
    ok = all(res.values())
    print(" ".join(f"{k}={v}" for k, v in res.items()), "applied_contact", ac)
    print("PARAM_CHAIN_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
