# Copyright 2026 The RLinf Authors.
"""Builder-only identifiability of grasp contact parameters at the holding frame."""
import hashlib,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cowm import roles
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SC=Path(__file__).parent;OUT=R/'grasp_ident_dev_v2';OUT.mkdir(exist_ok=False);EPS=[f'libero_goal_t{k}_i0' for k in range(6)];roles.require(EPS,{'dev'},'grasp identifiability development');sources={}
for ep in EPS:
    path=R/'grasp_contrast_dev_v1'/(ep+'__grasp')/'probe.json';probe=json.loads(path.read_text());stop=probe['phases']['holding'];probe['actions']=probe['actions'][:stop];probe['frames']=[f for f in probe['frames'] if f<=stop];dest=OUT/'probes'/ep;dest.mkdir(parents=True);(dest/'probe.json').write_text(json.dumps(probe,indent=1));sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
(OUT/'freeze.json').write_text(json.dumps({'episodes':EPS,'prefix':'holding frame240, before release','sources':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'Target_inputs':False},indent=1))
def run(param):
    args=[sys.executable,str(SC/'toolkits/cowm/s6/identifiability.py'),'--world',str(R/'bowl_support_dev_v1/world11'),'--state',str(R/'s6_natural_v3/shape_states'),'--probes',str(OUT/'probes'),'--out',str(OUT/(param+'.json')),'--params',param,'--episodes',','.join(EPS)]
    with (OUT/(param+'.log')).open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((param,r.returncode))
    return json.loads((OUT/(param+'.json')).read_text())
with ThreadPoolExecutor(2) as pool:rows=list(pool.map(run,['package.friction','package.density']))
report={}
for row in rows:report.update(row)
(OUT/'report.json').write_text(json.dumps(report,indent=1));print('GRASP_IDENTIFIABILITY',json.dumps(report),flush=True)
