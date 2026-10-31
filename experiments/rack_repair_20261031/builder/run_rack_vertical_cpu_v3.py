# Copyright 2026 The RLinf Authors.
"""Track only the robot through frozen public orthogonal-contact RGB, on CPU."""
import hashlib,json,shutil,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SC=Path(__file__).parent
PLAN=R/'rack_vertical_preflight_dev_v3';E=Path('/data_cf/cowmpo/evidence/codex_rack_vertical_dev_v3')
OUT=R/'rack_vertical_video_cpu_v3';READY=R/'rack_vertical_tracking_ready_v3';EP='libero_goal_t0_i0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
keys=json.loads((PLAN/'selection.json').read_text())['selected'];files={}
for key in keys:
    p=PLAN/key/'probe.json';query=json.loads((E/key/'query.json').read_text());probe=json.loads(p.read_text())
    if query['query_sha256']!=sha(p) or query['frames']!=probe['frames']:raise ValueError('query binding mismatch')
    files[str(p)]=sha(p)
    for step in probe['frames']:
        p=E/key/f'{EP}__f{step:03d}'/'frame0.png';files[str(p)]=sha(p)
for p,h in json.loads((R/'video_freeze_v1.json').read_text())['files'].items():
    if sha(p)!=h:raise ValueError('model/code changed')
READY.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False)
anchor=READY/'anchors'/EP;anchor.mkdir(parents=True)
source=R/'grasp_video_dev_v1/grasp'/f'{EP}__f000'/'masks.npz'
with np.load(source) as a:mask=a['robot']
np.savez_compressed(anchor/'masks.npz',robot=mask)
(anchor/'association.json').write_text(json.dumps({'episode':EP,'objects':{'robot':{'status':'matched','identity_source':'frozen first-frame RGB robot anchor'}},'masks_sha256':sha(anchor/'masks.npz'),'source_sha256':sha(source)},indent=1))
for key in keys:
    initial=READY/key/'initial'/EP;initial.mkdir(parents=True);shutil.copy2(E/key/f'{EP}__f000'/'frame0.png',initial/'frame0.png')
    probes=READY/key/'probes'/EP;probes.mkdir(parents=True);shutil.copy2(PLAN/key/'probe.json',probes/'probe.json')
frozen={'keys':keys,'files':files,'anchor_source_sha256':sha(source),'script_sha256':sha(__file__),'tracker_sha256':sha(SC/'track_probe_video_device.py'),'device':'cpu','GT_inputs':False}
(OUT/'freeze.json').write_text(json.dumps(frozen,indent=1))
def run(key):
    cmd=[sys.executable,str(SC/'track_probe_video_device.py'),'--data-role','dev','--device','cpu','--cpu-threads','8','--episodes',EP,'--anchors',str(READY/'anchors'),'--initial-evidence',str(READY/key/'initial'),'--evidence',str(E/key),'--probes',str(READY/key/'probes'),'--out',str(OUT/key)]
    with (OUT/(key+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((key,r.returncode))
    expected={f'{EP}__f{s:03d}' for s in json.loads((PLAN/key/'probe.json').read_text())['frames']}
    actual={p.parent.name for p in (OUT/key).glob('*/association.json')}
    if actual!=expected:raise ValueError('incomplete exact frame set')
    return {'key':key,'frames':len(actual)}
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,keys))
for p,h in files.items():
    if sha(p)!=h:raise ValueError('public evidence changed')
(OUT/'complete.json').write_text(json.dumps(rows,indent=1));print('RACK_MULTIDIR_CPU_COMPLETE',sum(r['frames'] for r in rows),flush=True)
