import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cowm import roles
SC=Path(__file__).parent
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
out=R/'state_no_contact_dev_v1';out.mkdir(exist_ok=False)
eps=sorted(p.name for p in (R/'visual_dev_bound_v1/dev3').iterdir() if p.is_dir())
roles.require(eps,{'dev'},'estimator contact ablation')
def run(i):
    part=eps[i::4]
    with (out/f'part{i}.log').open('w') as f:
        result=subprocess.run([sys.executable,str(SC/'estimate_no_contact.py'),
            '--world','/data_cf/cowmpo/worlds/s5_v4/shape/world_fit',
            '--evidence','/data_cf/cowmpo/evidence/libero_goal_dev3',
            '--masks',str(R/'visual_dev_bound_v1/dev3'),'--out',str(out/f'part{i}'),
            '--episodes',','.join(part)],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'part{i}: {result.returncode}')
    print('complete',i,flush=True)
with ThreadPoolExecutor(4) as pool:list(pool.map(run,range(4)))
print('ESTIMATOR_ABLATION_COMPLETE')
