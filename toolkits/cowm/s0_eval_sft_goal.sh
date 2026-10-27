#!/usr/bin/env bash
# S0: OpenVLA-OFT SFT (goal traj1) success rate on real LIBERO-goal, upstream RLinf eval protocol.
# Own Ray head (never joins another cluster), host LIBERO lock held for the whole run.
set -u
REPO=/mnt/nfs/fjj/Code/CoWMPO
V=/data_cf/cowmpo/.venv-oft-wan
MODEL=/mnt/nfs/fjj/Models/Openvla-oft-SFT-libero-goal-traj1
# Serial by default: few env subprocesses (each pins a CPU core), more rollout epochs.
# 50 envs x 10 epochs still covers all 500 goal trials (ordered eval pool, -1 once exhausted).
PLACE="${PLACE:-0}"
NENVS="${NENVS:-50}"
EPOCHS="${EPOCHS:-10}"
RAY_PORT=6461
RAY_TMP=/tmp/ray_cowmpo
OUT=/data_cf/cowmpo/s0/sft_goal_$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"

exec 9>/tmp/rlinf-libero-run.lock
if ! flock -n 9; then echo "LOCK_BUSY: another RLinf/LIBERO job owns the host lock"; exit 75; fi
export RLINF_LIBERO_RUN_LOCK_HELD=1

export PATH="$V/bin:$PATH" RAY_AUTH_MODE=disabled
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
if ray status --address="$RAY_ADDRESS" > /dev/null 2>&1; then
  echo "reusing CoWMPO ray head at $RAY_ADDRESS"
else
  ray start --head --port=$RAY_PORT --temp-dir=$RAY_TMP --include-dashboard=false \
    --num-cpus=64 --min-worker-port=30000 --max-worker-port=39999 > "$OUT/ray_start.log" 2>&1
  echo "ray head rc=$? at $RAY_ADDRESS"
fi

cd "$REPO"
bash evaluations/run_eval.sh libero libero_goal_openvlaoft_eval \
  "rollout.model.model_path=$MODEL" \
  "~cluster.component_placement" \
  "+cluster.component_placement.env=$PLACE" \
  "+cluster.component_placement.rollout=$PLACE" \
  "env.eval.total_num_envs=$NENVS" \
  "env.eval.rollout_epoch=$EPOCHS" > "$OUT/eval.log" 2>&1
rc=$?
# run_eval.sh writes its own log dir under $REPO/logs; record it next to our log
ls -dt "$REPO"/logs/*libero_goal_openvlaoft_eval 2>/dev/null | head -1 > "$OUT/rlinf_log_dir.txt"

# stop only this head (a bare `ray stop` would also kill other users' clusters)
pkill -f "temp-dir=$RAY_TMP" ; pkill -f "$RAY_TMP/session" ; sleep 3
echo "S0_EXIT rc=$rc out=$OUT"
