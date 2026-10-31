#!/usr/bin/env bash
# S6 runner. Stages: plan (Builder) | target (Target platform, host LIBERO lock) | perceive (Builder, GPU)
#                    | g2 (Builder)        Usage: run_s6.sh <stage> [world_dir] [tag]
set -eu
cd /mnt/nfs/fjj/Code/CoWMPO
STAGE=$1
WORLD=${2:-/data_cf/cowmpo/worlds/s4_ref_goal_v02}
TAG=${3:-ref}
SB=toolkits/cowm/isolation/run_builder.sh
STATE=${STATE:-/data_cf/cowmpo/worlds/s5_perception/dev3}
PROBES=/data_cf/cowmpo/worlds/s6_probe/dev3
EVID=/data_cf/cowmpo/evidence/libero_goal_dev3_probe
MASKS=/data_cf/cowmpo/worlds/s6_probe/dev3_masks
case $STAGE in
  plan)
    bash $SB python -m cowm.s6.plan_probe --world "$WORLD" --state "$STATE" --out $PROBES 2>&1 | grep -E "^libero|Error" ;;
  target)
    exec 9>/tmp/rlinf-libero-run.lock
    flock -n 9 || { echo LOCK_BUSY; exit 75; }
    cd toolkits && MUJOCO_EGL_DEVICE_ID=${EGL:-7} /data_cf/cowmpo/.venv-oft-wan/bin/python -m cowm.s6.probe_target_exec \
      --init-dir /data_cf/cowmpo/init/libero_goal_dev3 --probes $PROBES --evidence-out $EVID \
      --gt-out /data_cf/cowmpo/target_eval/s6_probe_gt/dev3 2>/dev/null | grep -E "^libero|DONE" ;;
  perceive)
    CUDA_VISIBLE_DEVICES=${GPU:-6} bash $SB /data_cf/cowmpo/.venv-perception/bin/python -m cowm.init.perceive \
      --world "$WORLD" --evidence $EVID --out $MASKS 2>&1 | grep -c "^libero" ;;
  g2)
    bash $SB python -m cowm.s6.g2_replay --world "$WORLD" --state "$STATE" --probes $PROBES --masks $MASKS \
      --out /data_cf/cowmpo/worlds/s6_probe/g2_${TAG}.json ${FRICTION:+--friction "$FRICTION"} 2>&1 | grep -E "^G2|Error" ;;
esac
echo S6_EXIT
