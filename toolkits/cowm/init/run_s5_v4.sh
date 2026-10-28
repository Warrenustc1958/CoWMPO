#!/usr/bin/env bash
# S5 v3 chain: perceive (prior + second pass) -> shape fit (warm start, ground-contact pose objective)
#              -> estimate all dev episodes -> Target-side GT comparison (host).
set -u
cd /mnt/nfs/fjj/Code/CoWMPO
SB=toolkits/cowm/isolation/run_builder.sh
EV=/data_cf/cowmpo/evidence/libero_goal_dev3
REF=/data_cf/cowmpo/worlds/s4_ref_goal_v02
OUT=/data_cf/cowmpo/worlds/s5_v4
PROG=/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/world/ref_goal_program_v03.py
FIT=$(for t in 0 1 2 3 4 5 6 7 8 9; do printf "libero_goal_t${t}_i0,"; done | sed "s/,$//")
mkdir -p $OUT
echo "== perceive $(date)"
CUDA_VISIBLE_DEVICES=6 bash $SB /data_cf/cowmpo/.venv-perception/bin/python -m cowm.init.perceive --world $REF \
  --evidence $EV --out $OUT/masks --prior /data_cf/cowmpo/worlds/s5_v2/prior 2>&1 | grep -c "^libero"
echo "== shape fit $(date)"
bash $SB python -m cowm.init.fit_shape --program $PROG --masks $OUT/masks --out $OUT/shape --fit $FIT --jobs 24 \
  --init /data_cf/cowmpo/worlds/s5_shape/shape_fit.json 2>&1 | grep --line-buffered -E "^(SHAPE_PARAMS|BASE|ROUND|SHAPE_DONE)|Traceback"
echo "== estimate $(date)"
bash $SB python -m cowm.init.estimate_state --world $OUT/shape/world_fit --evidence $EV --masks $OUT/masks \
  --out $OUT/dev3 2>&1 | grep -c "^libero"
for e in $OUT/masks/*/; do n=$(basename $e); cp $e/masks.npz $OUT/dev3/$n/ 2>/dev/null; done
cd toolkits && /data_cf/cowmpo/.venv-perception/bin/python -m cowm.init.s5_compare --est $OUT/dev3 \
  --gt /data_cf/cowmpo/target_eval/s5_gt/dev3 --out /data_cf/cowmpo/target_eval/s5_gt/report_v4_dev3.json
echo "S5V4_DONE $(date)"
