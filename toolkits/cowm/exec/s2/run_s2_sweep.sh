#!/usr/bin/env bash
# S2 sweep: LIBERO-goal tasks 0-9 x init 0-2, target vs Code WM executor (CPU only).
set -u
cd /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/exec/s2
P=/data_cf/cowmpo/.venv-oft-wan/bin/python
OUT=/data_cf/cowmpo/s2/sweep
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=""
for t in 0 1 2 3 4 5 6 7 8 9; do
  for i in 0 1 2; do
    tag="t${t}_i${i}"
    $P probe_target.py --task-id $t --init-id $i --out $OUT/target_$tag.npz > $OUT/log_$tag.txt 2>&1 \
      && $P probe_mirror.py --reset-from $OUT/target_$tag.npz --out $OUT/mirror_$tag.npz >> $OUT/log_$tag.txt 2>&1 \
      && $P compare.py --target $OUT/target_$tag.npz --mirror $OUT/mirror_$tag.npz --out $OUT/compare_$tag.json >> $OUT/log_$tag.txt 2>&1
    echo "$tag rc=$? $(grep -o 'WORST pos err mm: [0-9.]* [A-Z]*' $OUT/log_$tag.txt)"
  done
done
echo SWEEP_DONE
