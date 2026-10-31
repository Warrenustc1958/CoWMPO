#!/usr/bin/env bash
set -euo pipefail
R=/data_cf/cowmpo/worlds/codex_s56_20261030_1020
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4
run_builder() {
 env CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 COWM_PYTHONPATH="$SC/toolkits" bash /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh /data_cf/cowmpo/.venv-builder/bin/python "$@"
}
run_builder "$SC/fit_robot_rgb_remaining_dev.py"
run_builder "$SC/prepare_natural_validation.py"
run_builder "$SC/refit_natural_geometry_fine.py" --world "$R/bowl_support_dev_v1/world11" --object cream_cheese --masks "$R/natural_masks_remaining_dev_v1" --states "$R/stove_parts_states_dev_v1" --out "$R/natural_cheese_remaining_dev_v1" --role dev --jobs 8
run_builder "$SC/refit_natural_geometry_fine.py" --world "$R/bowl_support_dev_v1/world11" --object bowl --masks "$R/natural_masks_remaining_dev_v1" --states "$R/natural_cheese_remaining_dev_v1" --out "$R/natural_bowl_remaining_dev_v1" --role dev --jobs 8
run_builder "$SC/prepare_natural_validation.py" --after
run_builder "$SC/replay_natural_grasp.py" --world "$R/bowl_support_dev_v1/world11" --states "$R/natural_states_all_dev_v1" --robot "$R/robot_rgb_all_dev_v1" --probes "$R/grasp_contrast_dev_holdout_v1" --out "$R/grasp_natural_remaining_dev_v1" --expected 52 --jobs 8
