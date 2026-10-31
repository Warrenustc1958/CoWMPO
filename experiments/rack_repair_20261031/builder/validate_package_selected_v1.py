# Copyright 2026 The RLinf Authors.
"""Replay exact dev and fit grasp sets for a candidate already selected from fit RGB."""
import json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cowm.world.artifact import verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SC=Path(__file__).parent
best=json.loads((R/'package_contact_fit_v3/selection.json').read_text())['best'];verify_tree(best['world'],best['world_tree_sha256'])
cases=[('first6','dev',12,'natural_states_all_dev_v1','robot_rgb_all_dev_v2','grasp_contrast_dev_v1'),
       ('remaining','dev',52,'natural_states_all_dev_v1','robot_rgb_all_dev_v2','grasp_contrast_dev_holdout_v1'),
       ('fit','fit',56,'natural_bowl_fit_v2','robot_rgb_fit_v2','grasp_contrast_fit_v2')]
def run(case):
    name,role,count,states,robot,probes=case;out=R/('package_selected_'+name+'_v1')
    args=[sys.executable,str(SC/'replay_natural_grasp_role_v2.py'),'--role',role,'--expected',str(count),'--jobs','8','--world',best['world'],'--states',str(R/states),'--robot',str(R/robot),'--probes',str(R/probes),'--out',str(out)]
    with (R/('package_selected_'+name+'_v1.log')).open('w') as log:p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError((name,p.returncode))
    return {'set':name,'queries':count,'role':role,'output':str(out)}
with ThreadPoolExecutor(3) as pool:rows=list(pool.map(run,cases))
verify_tree(best['world'],best['world_tree_sha256']);(R/'package_selected_replay_v1.json').write_text(json.dumps(rows,indent=1));print(json.dumps(rows,indent=1),flush=True)
