"""Registry schema-2 tests (pure filesystem, no simulator). Exit code 1 on any failure.

  structural_fail_rejected : report ok=False with G0/G1 True (build_agent structural failure) -> rejected
  ok_executable            : ok + G0 + G1 -> executable, resolve works
  xml_change_refused       : edit an XML after registration -> resolve refuses
  g2_needs_pass            : promote to g2_verified with a failing summary -> refused
  no_skip                  : executable -> frozen directly -> refused
  ladder                   : executable -> g2_verified -> frozen with proper evidence -> ok
  min_level                : resolve(min_level="frozen") on an executable version -> refused
  rejected_not_promotable  : promoting a rejected version -> refused
"""
import json
import os
import sys
import tempfile

from cowm.world import registry as r


def world(d, name):
    w = os.path.join(d, name)
    os.makedirs(os.path.join(w, "assets"))
    json.dump({"world_sha256": name, "sdk_version": "t", "table": {}}, open(os.path.join(w, "world.json"), "w"))
    open(os.path.join(w, "assets", "bowl.xml"), "w").write("<mujoco/>")
    return w


def raises(fn):
    try:
        fn()
        return False
    except (ValueError, KeyError):
        return True


def main():
    res = {}
    gates = {"G0_build": True, "G1_self_consistent": True}
    with tempfile.TemporaryDirectory() as d:
        reg = os.path.join(d, "reg.json")
        a = r.register(world(d, "a"), report={"ok": False, "stage": "structural", "gates": gates}, path=reg)
        res["structural_fail_rejected"] = json.load(open(reg))["versions"][0]["status"] == "rejected" and \
            raises(lambda: r.resolve(a, path=reg))
        b = r.register(world(d, "b"), report={"ok": True, "gates": gates}, path=reg)
        res["ok_executable"] = r.resolve(b, path=reg).endswith("b")
        c = r.register(world(d, "c"), report={"ok": True, "gates": gates}, path=reg)
        open(os.path.join(d, "c", "assets", "bowl.xml"), "w").write("<mujoco><x/></mujoco>")
        res["xml_change_refused"] = raises(lambda: r.resolve(c, path=reg))
        res["g2_needs_pass"] = raises(lambda: r.promote(b, "g2_verified", {"G2_target_evidence": False}, path=reg))
        res["no_skip"] = raises(lambda: r.promote(b, "frozen", {"frozen_by": "x", "config": {}}, path=reg))
        res["min_level"] = raises(lambda: r.resolve(b, path=reg, min_level="frozen"))
        r.promote(b, "g2_verified", {"G2_target_evidence": True}, path=reg)
        r.promote(b, "frozen", {"frozen_by": "test", "config": {"sdk": "t"}}, path=reg)
        res["ladder"] = r.resolve(b, path=reg, min_level="frozen").endswith("b")
        res["rejected_not_promotable"] = raises(lambda: r.promote(a, "g2_verified", {"G2_target_evidence": True},
                                                                   path=reg))
    ok = all(res.values())
    print(" ".join(f"{k}={v}" for k, v in res.items()))
    print("REGISTRY_TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
