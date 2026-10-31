#!/usr/bin/env bash
# CPU test of run_rm_natural_v2.sh (no GPU, no policy): the launcher and validate_rm_run.py are copied next to
# a STUB rm_natural_control_v2.py that writes REAL npz files (or deliberately bad ones). Exit 1 on any mismatch.
#   ok         : 2x2 episodes, valid content                         -> 0
#   py_fail    : python exits 1                                      -> 1
#   sigkill    : python SIGKILLs itself                              -> 137 (+ time recorded)
#   short      : report says 3 episodes                              -> 3
#   missing    : only 3 npz                                          -> 3
#   corrupt    : one npz is not a numpy archive                      -> 4
#   wrong_id   : files named right, one file's content is another episode -> 4
#   bad_shape  : labels length != T+1 in one file                    -> 4
#   report_ids : report lists a trial that was not requested         -> 4
#   stale      : output dir already holds a previous report          -> 5 (stub never runs)
# usage: test_launcher.sh <path to run_rm_natural_v2.sh>
set -u
L=$1
D=$(dirname "$L")
# safety (10-30 incident): refuse a launcher that would not call the script next to itself (real policy)
grep -q '"$HERE/rm_natural_control_v2.py"' "$L" || { echo "REFUSED: $L does not call \$HERE/rm_natural_control_v2.py"; exit 2; }
[ -f "$D/validate_rm_run.py" ] || { echo "REFUSED: $D/validate_rm_run.py missing"; exit 2; }
PYV=${COWM_POLICY_PY:-/data_cf/cowmpo/.venv-oft-wan/bin/python}
T=$(mktemp -d /tmp/cowm_launcher_test.XXXXXX)
trap 'rm -rf "$T"' EXIT
cp "$L" $T/run_rm_natural_v2.sh
cp "$D/validate_rm_run.py" $T/validate_rm_run.py
cat > $T/rm_natural_control_v2.py <<'EOF'
import argparse, json, os, signal, sys
import numpy as np
p = argparse.ArgumentParser(); p.add_argument("--out"); p.add_argument("--tasks"); p.add_argument("--trials")
a = p.parse_args(); mode = os.environ["STUB_MODE"]
if mode == "py_fail": sys.exit(1)
if mode == "sigkill": os.kill(os.getpid(), signal.SIGKILL)
pairs = [(int(t), int(i)) for t in a.tasks.split(",") for i in a.trials.split(",")]
ed = os.path.join(a.out, "episodes"); os.makedirs(ed, exist_ok=True)
eps = []
for k, (t, i) in enumerate(pairs):
    if mode == "missing" and k == 3: continue
    T = 8
    lab = np.zeros(T + 1, bool); lab[5:] = True
    z = dict(actions=np.zeros((T, 7), np.float32), labels=lab, p=np.linspace(0, 1, T + 1),
             frames=np.zeros((3, 256, 256, 3), np.uint8), frame_idx=np.array([0, 4, 8]), task_id=t, trial=i)
    if mode == "bad_shape" and k == 1: z["labels"] = lab[:-1]
    if mode == "wrong_id" and k == 2: z["task_id"], z["trial"] = pairs[0]
    f = os.path.join(ed, f"t{t}_i{i}.npz")
    if mode == "corrupt" and k == 0:
        open(f, "w").write("x")
    else:
        np.savez_compressed(f, **z)
    eps.append({"task_id": t, "trial": (i + 100 if mode == "report_ids" and k == 0 else i), "first_success": 5})
n = 3 if mode == "short" else len(pairs)
json.dump({"overall": {"n_episodes": n}, "episodes": eps}, open(os.path.join(a.out, "report.json"), "w"))
EOF
fails=0
for case in ok:0 py_fail:1 sigkill:137 short:3 missing:3 corrupt:4 wrong_id:4 bad_shape:4 report_ids:4 stale:5; do
  m=${case%%:*}; want=${case#*:}
  if [ $m = stale ]; then mkdir -p $T/out_stale; echo '{}' > $T/out_stale/report.json; fi
  COWM_POLICY_PY=$PYV STUB_MODE=$m bash $T/run_rm_natural_v2.sh 0 $T/out_$m 0,1 0,1 $T/lock > $T/$m.log 2>&1; got=$?
  extra=""
  if [ $m = sigkill ]; then
    if grep -q "SIGKILL received" $T/out_$m/run_status.txt; then extra=" status_recorded"; else extra=" status_missing"; got=x; fi
  fi
  [ "$got" = "$want" ] && r=PASS || { r=FAIL; fails=$((fails + 1)); }
  printf "%-10s want %-4s got %-4s %s%s  (%s)\n" $m $want $got $r "$extra" "$(grep -E 'RUN_' $T/$m.log | tail -1)"
done
echo "LAUNCHER_TEST $([ $fails -eq 0 ] && echo PASS || echo FAIL)"
[ $fails -eq 0 ]
