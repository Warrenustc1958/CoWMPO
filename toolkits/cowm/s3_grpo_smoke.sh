#!/usr/bin/env bash
# CoWM-PO S3: tiny GRPO run in the Code WM backend (L0) on GPUs 6-7. Own Ray head, host LIBERO lock held.
set -u
REPO=/mnt/nfs/fjj/Code/CoWMPO
V=/data_cf/cowmpo/.venv-oft-wan
RAY_PORT=6461
RAY_TMP=/tmp/ray_cowmpo
OUT=/data_cf/cowmpo/s3/grpo_smoke_$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"

exec 9>/tmp/rlinf-libero-run.lock
if ! flock -n 9; then echo "LOCK_BUSY: another RLinf/LIBERO job owns the host lock"; exit 75; fi
export RLINF_LIBERO_RUN_LOCK_HELD=1

export PATH="$V/bin:$PATH" RAY_AUTH_MODE=disabled
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
if ray status --address="$RAY_ADDRESS" > /dev/null 2>&1; then
  echo "reusing CoWMPO ray head at $RAY_ADDRESS"
else
  ray start --head --port=$RAY_PORT --temp-dir=$RAY_TMP --include-dashboard=false \
    --num-cpus=48 --min-worker-port=30000 --max-worker-port=39999 > "$OUT/ray_start.log" 2>&1
  echo "ray head rc=$? at $RAY_ADDRESS"
fi

cd "$REPO"
bash examples/embodiment/run_embodiment.sh code_libero_goal_grpo_openvlaoft_smoke > "$OUT/train.log" 2>&1
rc=$?
ls -dt "$REPO"/logs/*code_libero_goal_grpo_openvlaoft_smoke 2>/dev/null | head -1 > "$OUT/rlinf_log_dir.txt"

# stop only this head (a bare `ray stop` would also kill other users' clusters)
pkill -f "temp-dir=$RAY_TMP" ; pkill -f "$RAY_TMP/session" ; sleep 3
echo "S3_EXIT rc=$rc out=$OUT"
