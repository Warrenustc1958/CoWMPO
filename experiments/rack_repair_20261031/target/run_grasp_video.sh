#!/usr/bin/env bash
set -euo pipefail
R=/data_cf/cowmpo/worlds/codex_s56_20261030_1020
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4
for mode in grasp miss; do
 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 COWM_PYTHONPATH="$SC/toolkits" bash /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh /data_cf/cowmpo/.venv-perception/bin/python "$SC/track_probe_video.py" --anchors "$R/grasp_tracking_ready_dev_v1/$mode/anchors" --initial-evidence "$R/grasp_tracking_ready_dev_v1/$mode/initial" --evidence "$R/grasp_tracking_ready_dev_v1/$mode/evidence" --probes "$R/grasp_tracking_ready_dev_v1/$mode/probes" --out "$R/grasp_video_dev_v1/$mode" --data-role dev
done
