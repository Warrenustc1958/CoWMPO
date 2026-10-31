#!/usr/bin/env bash
set -euo pipefail
R=/data_cf/cowmpo/worlds/codex_s56_20261030_1020
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v7
run_builder() {
 env CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 COWM_PYTHONPATH="$SC/toolkits" bash /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh /data_cf/cowmpo/.venv-builder/bin/python "$@"
}
run_builder "$SC/fit_robot_rgb_frozen.py" --world "$R/bowl_support_dev_v1/world11" --states "$R/stove_parts_states_fit_v1" --masks "$R/functional_complete_fit_v2" --evidence /data_cf/cowmpo/evidence/libero_goal_fit3 --out "$R/robot_rgb_fit_v2" --role fit --jobs 4
run_builder "$SC/refit_natural_geometry_v3.py" --world "$R/bowl_support_dev_v1/world11" --object cream_cheese --masks "$R/functional_complete_fit_v2" --states "$R/stove_parts_states_fit_v1" --robot "$R/robot_rgb_fit_v2" --out "$R/natural_cheese_fit_v2" --role fit --jobs 4
run_builder "$SC/refit_natural_geometry_v3.py" --world "$R/bowl_support_dev_v1/world11" --object bowl --masks "$R/functional_complete_fit_v2" --states "$R/natural_cheese_fit_v2" --robot "$R/robot_rgb_fit_v2" --out "$R/natural_bowl_fit_v2" --role fit --jobs 4
