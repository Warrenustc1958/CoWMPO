#!/usr/bin/env bash
# S7 RM natural control v2 (per-action success time, every frame scored, actions saved).
# Usage: run_rm_natural_v2.sh <gpu> <out> <tasks> <trials> [lock]
set -u
GPU=$1; OUT=$2; TASKS=$3; TRIALS=$4; LOCK=${5:-/tmp/rlinf-libero-run.lock}
mkdir -p "$OUT"
cd /data_cf/cowmpo/diagnostics
exec 9>"$LOCK"
flock -w 3600 9 || { echo LOCK_BUSY; exit 75; }
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  /data_cf/cowmpo/.venv-oft-wan/bin/python rm_natural_control_v2.py --out "$OUT" --tasks "$TASKS" --trials "$TRIALS" \
  2>&1 | grep -vE "WARNING|tensorflow|GetPrototype|cuda_(dnn|fft|blas)|TensorRT|oneDNN|AVX"
echo RUN_EXIT
