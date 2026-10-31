#!/usr/bin/env bash
set -euo pipefail
R=/data_cf/cowmpo/worlds/codex_s56_20261030_1020
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4
for key in h170 h270; do
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 COWM_PYTHONPATH="$SC/toolkits" bash /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh /data_cf/cowmpo/.venv-perception/bin/python "$SC/track_probe_video.py" --anchors "$R/rack_touch_tracking_ready_v1/$key/anchors" --initial-evidence "$R/rack_touch_tracking_ready_v1/$key/initial" --evidence "/data_cf/cowmpo/evidence/codex_rack_touch_dev_v1/$key" --probes "$R/rack_touch_dev_v1/$key/probes" --out "$R/rack_touch_video_v1/$key" --data-role dev
done
