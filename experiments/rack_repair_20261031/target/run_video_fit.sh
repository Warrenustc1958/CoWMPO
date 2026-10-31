#!/usr/bin/env bash
set -euo pipefail
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4
R=/data_cf/cowmpo/worlds/codex_s56_20261030_1020
SB=/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh
export COWM_PYTHONPATH="$SC/toolkits"
env CUDA_VISIBLE_DEVICES= bash "$SB" /data_cf/cowmpo/.venv-builder/bin/python "$SC/freeze_video.py"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 bash "$SB" /data_cf/cowmpo/.venv-perception/bin/python "$SC/track_probe_video.py" \
 --anchors "$R/s6_ready_v1/anchors/fit" --initial-evidence /data_cf/cowmpo/evidence/libero_goal_fit3 \
 --evidence /data_cf/cowmpo/evidence/libero_goal_fit3_strike --probes /data_cf/cowmpo/worlds/s6_fit3/probes \
 --out "$R/probe_video_fit_v1" --data-role fit
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 bash "$SB" /data_cf/cowmpo/.venv-perception/bin/python "$SC/track_probe_video.py" \
 --anchors "$R/s6_ready_v1/anchors/dev" --initial-evidence /data_cf/cowmpo/evidence/libero_goal_fit3 \
 --evidence /data_cf/cowmpo/evidence/libero_goal_fit3_strike --probes /data_cf/cowmpo/worlds/s6_fit3/probes \
 --out "$R/probe_video_exposed_v1" --data-role dev --episodes libero_goal_t1_i5,libero_goal_t5_i5
