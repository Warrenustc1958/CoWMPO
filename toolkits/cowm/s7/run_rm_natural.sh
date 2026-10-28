#!/usr/bin/env bash
# S7 RM natural control: OFT SFT rollouts in official LIBERO-Goal (fit trials), RM scored per frame.
# Usage: run_rm_natural.sh <gpu> <out> <tasks> <trials>
set -u
GPU=$1; OUT=$2; TASKS=$3; TRIALS=$4; LOCK=${5:-/tmp/rlinf-libero-run.lock}
mkdir -p "$OUT"
cd /data_cf/cowmpo/diagnostics
exec 9>"$LOCK"
flock -w 3600 9 || { echo LOCK_BUSY; exit 75; }
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  /data_cf/cowmpo/.venv-oft-wan/bin/python rm_natural_control.py --out "$OUT" --tasks "$TASKS" --trials "$TRIALS" \
  2>&1 | grep -vE "WARNING|tensorflow|GetPrototype|cuda_(dnn|fft|blas)|TensorRT|oneDNN|AVX"
echo RUN_EXIT
