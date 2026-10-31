#!/usr/bin/env bash
set -euo pipefail
for variant in interior highpass robust; do
 env CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONPATH=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4/toolkits /data_cf/cowmpo/.venv-builder/bin/python /data_cf/cowmpo/staging/codex_s56_20261030_1020/target_eval_robot_rgb_variants.py "$variant"
done
