"""S4 gate (run INSIDE the Builder sandbox), SDK 0.2: lint rejection, API rejection, full verifier on the
human reference program (compile, footprints, settle, stillness, determinism, reach, articulation, camera)."""
import json
import os
import sys
import tempfile

from cowm.world.check_program import check
from cowm.world.compile_world import compile_world
from cowm.world.lint import LintError, lint

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("COWM_WORLDS", "/data_cf/cowmpo/worlds")

BAD_LINT = {
    "import": "import os\ndef build(w):\n    pass\n",
    "open()": "def build(w):\n    open('/etc/passwd')\n",
    "private attr": "def build(w):\n    w._objects.append(1)\n",
    "dunder escape": "def build(w):\n    x = w.__class__\n",
    "timeline lambda": "def build(w):\n    f = lambda t: t\n",
    "two functions": "def build(w):\n    pass\ndef step(s):\n    pass\n",
    "while loop": "def build(w):\n    while True:\n        pass\n",
}
BAD_API = {
    "placement as param": "def build(w):\n    w.add('bowl', 'b', xy=(w.p('bx', 0.0, (-0.2, 0.2)), 0.0))\n",
    "raw friction": "def build(w):\n    w.add('bowl', 'b', xy=(0, 0), friction=0.9)\n",
    "set camera": "def build(w):\n    w.camera(pos=(1, 0, 1), quat=(1, 0, 0, 0))\n",
    "set table height": "def build(w):\n    w.table(size=(1, 1, 0.05), height=0.7)\n",
    "too-narrow bounds": "def build(w):\n    r = w.p('r', 0.07, (0.069, 0.071))\n",
    "unknown material": "def build(w):\n    m = w.material('m', 'unobtainium')\n",
}


def main():
    ok = True
    for name, src in BAD_LINT.items():
        try:
            lint(src)
            good = False
        except (LintError, SyntaxError):
            good = True
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] lint rejects {name}")
    for name, src in BAD_API.items():
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=OUT, delete=False) as f:
            f.write(src)
        try:
            compile_world(f.name, os.path.join(OUT, "_api_reject"))
            good, why = False, "accepted"
        except ValueError as e:
            good, why = True, str(e)[:90]
        os.remove(f.name)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] API rejects {name}: {why}")

    out = os.path.join(OUT, "s4_ref_goal_v02")
    rep = check(os.path.join(HERE, "ref_goal_program.py"), out)
    with open(os.path.join(out, "check_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    ok &= rep["ok"]
    print(f"[{'PASS' if rep['ok'] else 'FAIL'}] reference program full verifier: stage={rep['stage']} "
          f"errors={rep['errors']}")
    u = rep.get("usability", {})
    print("   reach (m):", u.get("reach_dist_m"))
    print("   articulation (range frac):", u.get("articulation_range_frac"))
    print("   image px:", u.get("image_px"))
    print("   params:", rep.get("world", {}).get("params"))
    print("   render:", rep.get("render", rep.get("render_error")))
    print("S4_SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
