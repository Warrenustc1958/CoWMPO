#!/usr/bin/env bash
set -euo pipefail
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v7
run_builder() {
 env CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 COWM_PYTHONPATH="$SC/toolkits" bash /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh /data_cf/cowmpo/.venv-builder/bin/python "$@"
}
run_builder "$SC/correct_robot_integrator.py"
run_builder "$SC/run_contract_tests.py"
run_builder "$SC/test_robot_replay.py"
R=/data_cf/cowmpo/worlds/codex_s56_20261030_1020
run_builder "$SC/replay_natural_grasp.py" --world "$R/bowl_support_dev_v1/world11" --states "$R/natural_states_all_dev_v1" --robot "$R/robot_rgb_all_dev_v2" --probes "$R/grasp_contrast_dev_v1" --out "$R/grasp_natural_first6_dev_v2" --expected 12 --jobs 8
run_builder "$SC/replay_natural_grasp.py" --world "$R/bowl_support_dev_v1/world11" --states "$R/natural_states_all_dev_v1" --robot "$R/robot_rgb_all_dev_v2" --probes "$R/grasp_contrast_dev_holdout_v1" --out "$R/grasp_natural_remaining_dev_v2" --expected 52 --jobs 8
