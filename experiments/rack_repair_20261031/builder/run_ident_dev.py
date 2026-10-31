"""Dev-only sensitivity ablation: identical frozen geometry/state/actions, alternate contact mixing."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cowm import roles
from cowm.world.artifact import tree_sha

SC=Path(__file__).parent
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
world=Path('/data_cf/cowmpo/worlds/s5_v4/shape/world_fit')
pair=R/'world_pair_v1';shutil.copytree(world,pair)
w=json.loads((pair/'world.json').read_text());w.pop('world_sha256')
w['table']['contact_rule']='object_table_pair'
w['contact_ablation_note']='same frozen meshes; only material contact mixing differs'
w['world_sha256']=hashlib.sha256(json.dumps(w,sort_keys=True).encode()).hexdigest()
(pair/'world.json').write_text(json.dumps(w,indent=2))
eps=['libero_goal_t0_i0','libero_goal_t1_i0','libero_goal_t2_i0','libero_goal_t3_i0','libero_goal_t4_i0','libero_goal_t5_i0']
roles.require(eps,{'dev'},'dev identifiability ablation')
out=R/'ident_dev_v1';out.mkdir(exist_ok=False)
def run(item):
    name,path=item
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.run([sys.executable,str(SC/'toolkits/cowm/s6/identifiability.py'),
            '--world',str(path),'--state',str(R/'state_visual_dev_merged_v1/dev3'),
            '--probes','/data_cf/cowmpo/worlds/s6_strike/dev3','--out',str(out/(name+'.json')),
            '--episodes',','.join(eps)],stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError((name,p.returncode))
    print(name,json.loads((out/(name+'.json')).read_text()),flush=True)
before={str(p):tree_sha(p) for p in [world,pair]}
with ThreadPoolExecutor(2) as pool:list(pool.map(run,[('legacy',world),('pair',pair)]))
if before!={str(p):tree_sha(p) for p in [world,pair]}:raise RuntimeError('world mutated')
(out/'provenance.json').write_text(json.dumps({'roles':'dev only','episodes':eps,'world_hashes':before},indent=2))
print('IDENT_DEV_COMPLETE')
