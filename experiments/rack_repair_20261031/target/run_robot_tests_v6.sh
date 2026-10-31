#!/usr/bin/env bash
set -euo pipefail
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v6
for name in run_contract_tests.py test_robot_replay.py; do
 env CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 COWM_PYTHONPATH="$SC/toolkits" bash /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh /data_cf/cowmpo/.venv-builder/bin/python "$SC/$name"
done
