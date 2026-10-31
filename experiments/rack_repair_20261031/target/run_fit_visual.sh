#!/usr/bin/env bash
set -euo pipefail
S=/data_cf/cowmpo/staging/codex_s56_20261030_1020
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020
O=/data_cf/cowmpo/worlds/codex_s56_20261030_1020
SB=/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh
export COWM_PYTHONPATH="$SC/toolkits"
P=/data_cf/cowmpo/.venv-perception/bin/python
env CUDA_VISIBLE_DEVICES= bash "$SB" /data_cf/cowmpo/.venv-builder/bin/python "$SC/freeze_visual.py"
EPS=$(env PYTHONPATH="$SC/toolkits" /data_cf/cowmpo/.venv-builder/bin/python -c 'from cowm import roles;from pathlib import Path;print(",".join(roles.select(sorted(p.name for p in Path("/data_cf/cowmpo/worlds/s5_audit_v1/fit3").iterdir() if p.is_dir()),"fit")))')
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 bash "$SB" "$P" "$SC/extract_visual_features.py" \
 --pool /data_cf/cowmpo/worlds/s5_audit_v1/fit3 --evidence /data_cf/cowmpo/evidence/libero_goal_fit3 \
 --world /data_cf/cowmpo/worlds/s4_ref_goal_v02 --out "$O/visual_fit_v1/fit3" --data-role fit --episodes "$EPS" \
 > "$S/visual_fit_features.log" 2>&1
env CUDA_VISIBLE_DEVICES= bash "$SB" /data_cf/cowmpo/.venv-builder/bin/python "$SC/run_visual_association.py" \
 --features "$O/visual_fit_v1/fit3" --pool /data_cf/cowmpo/worlds/s5_audit_v1/fit3 \
 --bank "$O/visual_dev_analysis_v1/report.npz" --freeze "$O/visual_freeze_v1.json" \
 --prior /data_cf/cowmpo/worlds/s5_v2/prior --out "$O/visual_fit_result_v1/fit3" --data-role fit
echo FIT_VISUAL_COMPLETE
