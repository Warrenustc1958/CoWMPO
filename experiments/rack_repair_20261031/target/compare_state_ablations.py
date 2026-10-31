import shutil
import subprocess
import sys
from pathlib import Path
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
SC=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v2')
dest=R/'state_no_contact_dev_merged_v1';dest.mkdir(exist_ok=False)
for p in (R/'state_no_contact_dev_v1').glob('part*/libero_goal_*/state.json'):
    d=dest/p.parent.name;shutil.copytree(p.parent,d)
    shutil.copy2(R/'visual_dev_bound_v1/dev3'/p.parent.name/'masks.npz',d/'masks.npz')
for est,gt,label in [(dest,'dev3','no_contact_dev'),(R/'state_fit_visual_v3','fit3','fit_visual')]:
    if label=='fit_visual':
        for p in est.glob('libero_goal_*/state.json'):
            shutil.copy2(R/'visual_fit_result_v3/fit3'/p.parent.name/'masks.npz',p.parent/'masks.npz')
    subprocess.run([sys.executable,str(SC/'toolkits/cowm/init/s5_compare.py'),'--est',str(est),
      '--gt','/data_cf/cowmpo/target_eval/s5_gt/'+gt,
      '--out','/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/state_'+label+'.json'],check=True)
