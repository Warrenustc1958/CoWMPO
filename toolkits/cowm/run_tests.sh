#!/usr/bin/env bash
# CoWM-PO CPU regression suite (OSMesa). Runs every suite, records each exit code, prints a summary and
# exits 1 if ANY suite failed (no trailing echo can mask a failure). Full log + summary go to $LOGDIR.
# Usage: toolkits/cowm/run_tests.sh [logdir]      (run from anywhere; uses the repo this script lives in)
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TK=$(dirname "$HERE"); REPO=${COWM_REPO:-$(dirname "$TK")}   # COWM_REPO: tests/ + rlinf backend for staging runs
LOGDIR=${1:-/data_cf/cowmpo/test_logs/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=$TK
export COWM_TEST_WORLD_DIR=${COWM_TEST_WORLD_DIR:-/data_cf/cowmpo/worlds/s6_ident_v2/world_r050}
export COWM_TEST_OTHER_WORLD_DIR=${COWM_TEST_OTHER_WORLD_DIR:-/data_cf/cowmpo/worlds/s4_ref_goal_v02}
PY=${COWM_BUILDER_PY:-/data_cf/cowmpo/.venv-builder/bin/python}
PYW=${COWM_POLICY_PY:-/data_cf/cowmpo/.venv-oft-wan/bin/python}
W5=/data_cf/cowmpo/worlds/s5_v4/shape/world_fit; ST5=/data_cf/cowmpo/worlds/s5_v4/dev3
PR=/data_cf/cowmpo/worlds/s6_strike/dev3; REF=/data_cf/cowmpo/worlds/s4_ref_goal_v02
cd "$TK"
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null); DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)
{ echo "repo $REPO"; echo "commit $COMMIT dirty_files $DIRTY"; echo "start $(date -Is)"; } > "$LOGDIR/summary.txt"
declare -a NAMES CODES
suite() {   # name, command...
  local name=$1; shift
  echo "==== $name $(date +%T)" >> "$LOGDIR/full.log"
  "$@" >> "$LOGDIR/full.log" 2>&1
  local rc=$?
  NAMES+=("$name"); CODES+=("$rc")
  printf "%-22s %s\n" "$name" "$([ $rc -eq 0 ] && echo PASS || echo "FAIL($rc)")" | tee -a "$LOGDIR/summary.txt"
}
suite world_env        $PYW -m pytest -q -p no:cacheprovider "$REPO/tests/cowm/test_world_env.py"
suite snapshot         $PY -m cowm.world.test_snapshot $W5 $REF
suite snapshot_contact $PY cowm/world/test_snapshot_contact.py $W5 $ST5 $PR libero_goal_t0_i0
suite reset_order      $PY cowm/world/test_reset_order.py $W5 $ST5 libero_goal_t0_i0 libero_goal_t5_i1
suite batch_partition  $PYW cowm/world/test_batch_partition.py "$REPO" "$REPO/rlinf/envs/sim/world_model/backend/code.py" \
                       $W5 $ST5 libero_goal_t0_i0,libero_goal_t1_i0,libero_goal_t5_i1,libero_goal_t8_i2
suite contact_rule     $PY -m cowm.world.test_contact_rule "$LOGDIR/contact"
suite param_chain      $PYW cowm/world/test_param_chain.py "$REPO" "$REPO/rlinf/envs/sim/world_model/backend/code.py" "$LOGDIR/chain"
suite g2_gate          $PY -m cowm.s6.test_g2_gate
suite registry         $PY -m cowm.world.test_registry
suite roles            $PY -m cowm.test_roles
FAILS=0; for c in "${CODES[@]}"; do [ "$c" -eq 0 ] || FAILS=$((FAILS + 1)); done
echo "end $(date -Is) suites ${#CODES[@]} failed $FAILS" | tee -a "$LOGDIR/summary.txt"
echo "logs: $LOGDIR/full.log $LOGDIR/summary.txt"
[ $FAILS -eq 0 ]
