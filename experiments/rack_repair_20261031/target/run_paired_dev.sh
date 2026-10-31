#!/usr/bin/env bash
set -euo pipefail
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v2
R=/data_cf/cowmpo/worlds/codex_s56_20261030_1020
SB=/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh
export COWM_PYTHONPATH="$SC/toolkits"
for DATA in dev3 fit3; do
 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 bash "$SB" /data_cf/cowmpo/.venv-perception/bin/python "$SC/paired_resolve.py" \
  --source "$R/paired_dev_source_v3/$DATA" --out "$R/paired_dev_result_v3/$DATA" \
  --pool "/data_cf/cowmpo/worlds/s5_audit_v1/$DATA" --evidence "/data_cf/cowmpo/evidence/libero_goal_$DATA" \
  --world /data_cf/cowmpo/worlds/s4_ref_goal_v02 --freeze "$R/visual_freeze_v3_paired.json" --data-role dev
done
env CUDA_VISIBLE_DEVICES= PYTHONPATH="$SC/toolkits" /data_cf/cowmpo/.venv-oft-wan/bin/python /data_cf/cowmpo/staging/p6_v1/recheck_eval.py \
 --pool-root /data_cf/cowmpo/worlds/s5_audit_v1 --recheck-root "$R/paired_dev_result_v3" \
 --gt-root /data_cf/cowmpo/target_eval/s5_gt --out /data_cf/cowmpo/target_eval/codex_s56_20261030_1020/paired_dev_v3.json
