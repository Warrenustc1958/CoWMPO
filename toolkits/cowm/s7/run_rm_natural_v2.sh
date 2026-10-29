#!/usr/bin/env bash
# S7 RM natural control v2 (per-action success time, every frame scored, actions saved).
# Usage: run_rm_natural_v2.sh <gpu> <out> <tasks> <trials> [lock]
# Exit code = 0 only if python exited 0 AND report.json lists tasks x trials episodes AND every episode npz
# exists; otherwise non-zero with the reason printed (audit 10.29: no trailing echo may mask a failure).
# Launch long runs detached (setsid nohup ...); that is an operational measure only: the SIGKILL (exit 137)
# seen on 10-29 has no identified cause yet (no kernel OOM, no cgroup limit, ssh session was still alive).
set -uo pipefail
GPU=$1; OUT=$2; TASKS=$3; TRIALS=$4; LOCK=${5:-/tmp/rlinf-libero-run.lock}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT"
exec 9>"$LOCK"
flock -w 3600 9 || { echo "RUN_FAILED lock busy: $LOCK"; exit 75; }
echo "start $(date -Is) host $(hostname) pid $$ gpu $GPU tasks $TASKS trials $TRIALS" > "$OUT/run_status.txt"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  /data_cf/cowmpo/.venv-oft-wan/bin/python "$HERE/rm_natural_control_v2.py" --out "$OUT" --tasks "$TASKS" \
  --trials "$TRIALS" 2>&1 | grep --line-buffered -vE "WARNING|tensorflow|GetPrototype|cuda_(dnn|fft|blas)|TensorRT|oneDNN|AVX"
rc=${PIPESTATUS[0]}
echo "python_exit $rc $(date -Is)" >> "$OUT/run_status.txt"
if [ "$rc" -ne 0 ]; then
  [ "$rc" -eq 137 ] && echo "SIGKILL received: record time $(date -Is), check journalctl/dmesg now" >> "$OUT/run_status.txt"
  echo "RUN_FAILED python exit $rc"; exit "$rc"
fi
nt=$(echo "$TASKS" | tr ',' '\n' | grep -c .); ntr=$(echo "$TRIALS" | tr ',' '\n' | grep -c .); want=$((nt * ntr))
got=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['overall']['n_episodes'])" "$OUT/report.json" 2>/dev/null || echo 0)
files=$(ls "$OUT"/episodes/*.npz 2>/dev/null | wc -l)
echo "episodes report $got files $files expected $want" >> "$OUT/run_status.txt"
if [ "$got" -ne "$want" ] || [ "$files" -ne "$want" ]; then
  echo "RUN_FAILED incomplete: report $got files $files expected $want"; exit 3
fi
echo "RUN_OK $want episodes $(date -Is)" | tee -a "$OUT/run_status.txt"
