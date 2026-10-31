import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
SC=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020')
for dataset in ['dev3','fit3']:
    dest=ROOT/'state_visual_dev_merged_v1'/dataset
    dest.mkdir(parents=True,exist_ok=False)
    for source in sorted((ROOT/'state_visual_dev_v1').glob(dataset+'_part*')):
        if not source.is_dir():continue
        for p in sorted(source.glob('*/state.json')):
            target=dest/p.parent.name
            shutil.copytree(p.parent,target)
            shutil.copy2(ROOT/'visual_dev_bound_v1'/dataset/p.parent.name/'masks.npz',target/'masks.npz')
    subprocess.run([sys.executable,str(SC/'toolkits/cowm/init/s5_compare.py'),
       '--est',str(dest),'--gt','/data_cf/cowmpo/target_eval/s5_gt/'+dataset,
       '--out','/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/state_visual_'+dataset+'_v1.json'],check=True)
