#!/usr/bin/env bash
set -euo pipefail
R=/data_cf/cowmpo/worlds/codex_s56_20261030_1020
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4
for combo in original_edges support_edges support_basic; do
 world="$R/stove_parts_geometry_dev_v1/world07"; states="$R/stove_parts_states_dev_v1"; robot="$R/robot_rgb_edges_dev_v1"
 if [[ "$combo" == support_* ]]; then world="$R/cheese_rgb_support_dev_v2/world"; states="$R/cheese_rgb_support_dev_v2/states"; fi
 if [[ "$combo" == support_basic ]]; then robot="$R/robot_rgb_dev_v1"; fi
 env CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 COWM_PYTHONPATH="$SC/toolkits" bash /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh /data_cf/cowmpo/.venv-builder/bin/python "$SC/replay_natural_grasp.py" --world "$world" --states "$states" --robot "$robot" --probes "$R/grasp_contrast_dev_v1" --out "$R/grasp_rgb_${combo}_dev_v1" --expected 12 --jobs 6
done
