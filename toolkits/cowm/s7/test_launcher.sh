#!/usr/bin/env bash
# CPU test of run_rm_natural_v2.sh exit semantics (no GPU, no policy): the launcher is copied next to a STUB
# rm_natural_control_v2.py that simulates each outcome. Exit 1 if any case gives the wrong exit code.
#   ok        : python 0, report 2x2=4 episodes, 4 npz      -> 0
#   py_fail   : python exits 1                              -> 1
#   sigkill   : python SIGKILLs itself                      -> 137, run_status.txt records the time
#   short     : python 0, report says 3 episodes            -> 3
#   missing   : python 0, report 4 episodes, only 3 npz     -> 3
# usage: test_launcher.sh <path to run_rm_natural_v2.sh>
set -u
L=$1
# safety (10-30 incident): a launcher that does not call the python script next to itself would start the
# REAL policy instead of the stub. Refuse to run it.
grep -q '"$HERE/rm_natural_control_v2.py"' "$L" || { echo "REFUSED: $L does not call \$HERE/rm_natural_control_v2.py"; exit 2; }
T=$(mktemp -d /tmp/cowm_launcher_test.XXXXXX)
trap 'rm -rf "$T"' EXIT
cp "$L" $T/run_rm_natural_v2.sh
cat > $T/rm_natural_control_v2.py <<'EOF'
import argparse, json, os, signal, sys
p = argparse.ArgumentParser(); p.add_argument("--out"); p.add_argument("--tasks"); p.add_argument("--trials")
a = p.parse_args(); mode = os.environ["STUB_MODE"]
if mode == "py_fail": sys.exit(1)
if mode == "sigkill": os.kill(os.getpid(), signal.SIGKILL)
os.makedirs(os.path.join(a.out, "episodes"), exist_ok=True)
n = 3 if mode == "short" else 4
json.dump({"overall": {"n_episodes": n}}, open(os.path.join(a.out, "report.json"), "w"))
for i in range(3 if mode == "missing" else n):
    open(os.path.join(a.out, "episodes", f"e{i}.npz"), "w").write("x")
EOF
fails=0
for case in ok:0 py_fail:1 sigkill:137 short:3 missing:3; do
  m=${case%%:*}; want=${case#*:}
  STUB_MODE=$m bash $T/run_rm_natural_v2.sh 0 $T/out_$m 0,1 0,1 $T/lock > $T/$m.log 2>&1; got=$?
  extra=""
  [ $m = sigkill ] && { grep -q "SIGKILL received" $T/out_$m/run_status.txt && extra=" status_recorded" || { extra=" status_missing"; got=x; }; }
  [ "$got" = "$want" ] && r=PASS || { r=FAIL; fails=$((fails + 1)); }
  printf "%-8s want %-4s got %-4s %s%s  (%s)\n" $m $want $got $r "$extra" "$(tail -1 $T/$m.log)"
done
echo "LAUNCHER_TEST $([ $fails -eq 0 ] && echo PASS || echo FAIL)"
[ $fails -eq 0 ]
