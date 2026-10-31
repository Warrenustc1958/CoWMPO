#!/usr/bin/env bash
set -euo pipefail
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4
for variant in interior highpass robust; do
 env CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 COWM_PYTHONPATH="$SC/toolkits" bash /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh /data_cf/cowmpo/.venv-builder/bin/python "$SC/fit_robot_rgb_$variant.py"
done
