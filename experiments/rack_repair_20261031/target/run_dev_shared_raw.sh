#!/usr/bin/env bash
set -euo pipefail
S=/data_cf/cowmpo/staging/codex_s56_20261030_1020
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020
O=/data_cf/cowmpo/worlds/codex_s56_20261030_1020/paired_raw_dev_v1
SB=/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh
test -d "$SC"
true
export COWM_PYTHONPATH="$SC/toolkits"
for dataset in dev3 fit3; do
  if [ "$dataset" = dev3 ]; then
    EV=libero_goal_dev3
    EPS=$(ls /data_cf/cowmpo/worlds/s5_audit_v1/dev3 | paste -sd,)
  else
    EV=libero_goal_fit3
    EPS=libero_goal_t1_i5,libero_goal_t5_i5
  fi
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 bash "$SB" \
    /data_cf/cowmpo/.venv-perception/bin/python "$SC/toolkits/cowm/init/run_recheck.py" \
    --strategy paired-audit --raw-scores --pool /data_cf/cowmpo/worlds/s5_audit_v1/"$dataset" \
    --evidence /data_cf/cowmpo/evidence/"$EV" --world /data_cf/cowmpo/worlds/s4_ref_goal_v02 \
    --prior /data_cf/cowmpo/worlds/s5_v2/prior --out "$O/$dataset" --episodes "$EPS" \
    > "$S/paired_raw_$dataset.log" 2>&1
  tail -3 "$S/paired_raw_$dataset.log"
done
echo PAIRED_DEV_COMPLETE
