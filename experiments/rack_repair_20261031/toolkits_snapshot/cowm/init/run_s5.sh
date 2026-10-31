#!/usr/bin/env bash
# S5 runner: EvidenceAdapter (GPU, perception venv) then InitialStateEstimate (CPU, builder venv),
# both inside the Builder sandbox.  Usage: run_s5.sh <stage: perceive|estimate|all> [episodes]
set -eu
cd /mnt/nfs/fjj/Code/CoWMPO
STAGE=${1:-all}
EPS=${2:-}
WORLD=${WORLD:-/data_cf/cowmpo/worlds/s4_ref_goal_v02}
EVID=/data_cf/cowmpo/evidence/libero_goal_dev3
OUT=${OUT:-/data_cf/cowmpo/worlds/s5_perception/dev3}
SB=toolkits/cowm/isolation/run_builder.sh
if [ "$STAGE" = perceive ] || [ "$STAGE" = all ]; then
  CUDA_VISIBLE_DEVICES=${GPU:-6} bash $SB /data_cf/cowmpo/.venv-perception/bin/python -m cowm.init.perceive \
    --world "$WORLD" --evidence "$EVID" --out "$OUT" ${EPS:+--episodes "$EPS"} 2>&1 | grep -E "^libero|^device|Error|error" || true
fi
if [ "$STAGE" = estimate ] || [ "$STAGE" = all ]; then
  bash $SB python -m cowm.init.estimate_state --world "$WORLD" --evidence "$EVID" --masks "$OUT" \
    --out "$OUT" ${EPS:+--episodes "$EPS"} 2>&1 | grep -v -i "warn\|macro" || true
fi
echo S5_EXIT
