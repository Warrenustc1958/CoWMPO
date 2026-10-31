#!/usr/bin/env bash
set -euo pipefail
SC=/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4
R=/data_cf/cowmpo/worlds/codex_s56_20261030_1020/s6_ready_v1
SB=/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/isolation/run_builder.sh
export COWM_PYTHONPATH="$SC/toolkits"
export CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
bash "$SB" /data_cf/cowmpo/.venv-builder/bin/python "$SC/prepare_s6_inputs.py"
for ROLE in dev fit; do
 bash "$SB" /data_cf/cowmpo/.venv-builder/bin/python "$SC/toolkits/cowm/init/batch_estimate.py" \
  --world "$R/reference_world_pair" --masks "$R/anchors/$ROLE" --evidence "$R/evidence" \
  --out "$R/reference_states_$ROLE" --data-role "$ROLE" --jobs 6
done
