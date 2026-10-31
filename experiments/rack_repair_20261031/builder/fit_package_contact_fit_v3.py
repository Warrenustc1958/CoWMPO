# Copyright 2026 The RLinf Authors.
"""Choose frozen contact candidates using all fit RGB responses, with no GT labels."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm import roles
from cowm.init.estimate_state import Scene, iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree

R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'package_contact_fit_v3'
VIDEO=R/'grasp_video_fit_cpu_v2/frames'
STATES=R/'natural_bowl_fit_v2'
ROBOTS=R/'robot_rgb_fit_v2'
PROBES=R/'grasp_contrast_fit_v2'
STEPS=list(range(190,241,10))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(index):
    frozen=json.loads((OUT/'freeze.json').read_text())
    spec=frozen['candidates'][index];world=Path(spec['world']);verify_tree(world,spec['tree_sha256'])
    scene=Scene(str(world));k=scene.names.index('cream_cheese');rows=[]
    for ep in frozen['episodes']:
        scene.reset();doc=json.loads((STATES/ep/'state.json').read_text())
        apply_state(scene,doc['objects'],require_matched=True,robot=json.loads((ROBOTS/ep/'robot.json').read_text())['robot'])
        probe=json.loads((PROBES/(ep+'__grasp')/'probe.json').read_text());values=[];statuses=[]
        for step,act in enumerate(probe['actions'],1):
            scene.env.step(np.asarray(act))
            if step not in STEPS:continue
            frame=VIDEO/f'{ep}__f{step:03d}';rec=json.loads((frame/'association.json').read_text())
            if rec['episode']!=frame.name or rec['masks_sha256']!=sha(frame/'masks.npz'):raise ValueError('video binding mismatch')
            status=rec['objects']['cream_cheese']['status'];statuses.append(status)
            if status=='matched':
                with np.load(frame/'masks.npz') as data:
                    target=data['cream_cheese']
                    if target.dtype!=bool or target.shape!=(256,256) or not target.any():raise ValueError('invalid matched mask')
                values.append(iou(scene.seg()==k,target))
            else:values.append(0.)
        if len(values)!=len(STEPS):raise ValueError('response frame set incomplete')
        rows.append({'episode':ep,'mean_response_iou':float(np.mean(values)),'values':values,'statuses':statuses})
    scene.env.close();verify_tree(world,spec['tree_sha256'])
    report={'candidate':index,'radius_m':spec['radius_m'],'world':str(world),'world_tree_sha256':spec['tree_sha256'],
            'mean_response_iou':float(np.mean([r['mean_response_iou'] for r in rows])),
            'episodes':len(rows),'expected_frames':len(rows)*len(STEPS),
            'matched_frames':sum(s=='matched' for r in rows for s in r['statuses']),'rows':rows}
    (OUT/f'candidate{index:02d}.json').write_text(json.dumps(report,indent=1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int);a=p.parse_args()
    if a.index is not None:return worker(a.index)
    OUT.mkdir(exist_ok=False)
    complete=json.loads((VIDEO.parent/'complete.json').read_text())
    if complete['episodes']!=28 or complete['frames']!=840:raise ValueError('fit video incomplete')
    eps=sorted(q.name for q in STATES.iterdir() if (q/'state.json').exists());roles.require(eps,{'fit'},'package fit')
    if len(eps)!=28:raise ValueError('wrong fit set')
    files={};candidates=[];matched=0
    for ep in eps:
        for q in [STATES/ep/'state.json',ROBOTS/ep/'robot.json',PROBES/(ep+'__grasp')/'probe.json']:files[str(q)]=sha(q)
        probe=json.loads((PROBES/(ep+'__grasp')/'probe.json').read_text())
        if not set(STEPS)<=set(probe['frames']):raise ValueError('requested response frame absent')
        for step in STEPS:
            frame=VIDEO/f'{ep}__f{step:03d}'
            for q in [frame/'masks.npz',frame/'association.json']:files[str(q)]=sha(q)
            matched+=json.loads((frame/'association.json').read_text())['objects']['cream_cheese']['status']=='matched'
    for index,radius in enumerate([0.,.002,.004,.006]):
        world=R/'package_contact_dev_v2'/f'world{index:02d}'
        candidates.append({'world':str(world),'tree_sha256':tree_sha(world),'radius_m':radius})
    frozen={'episodes':eps,'role':'fit','candidates':candidates,'steps':STEPS,'files':files,
            'script_sha256':sha(__file__),'objective':'all 28 fit equally weighted, observed cream_cheese response IoU steps190..240; unmatched scores zero; no sample deletion',
            'matched_frames':matched,'expected_frames':len(eps)*len(STEPS),'GT_inputs':False,
            'qualification':'adaptive fit; contact family from dev; physical labels not used for selection'}
    (OUT/'freeze.json').write_text(json.dumps(frozen,indent=1))
    def run(i):
        with (OUT/f'candidate{i:02d}.log').open('w') as log:r=subprocess.run([sys.executable,__file__,'--index',str(i)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(('candidate failed',i,r.returncode))
        return json.loads((OUT/f'candidate{i:02d}.json').read_text())
    with ThreadPoolExecutor(4) as pool:reports=list(pool.map(run,range(4)))
    for q,h in files.items():
        if sha(q)!=h:raise ValueError('fit input changed')
    best=max(reports,key=lambda r:(r['mean_response_iou'],-r['candidate']))
    brief=lambda r:{k:v for k,v in r.items() if k!='rows'}
    selection={'best':brief(best),'candidates':[brief(r) for r in reports],'all_frozen_inputs_unchanged':True}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=1));print(json.dumps(selection,indent=1),flush=True)
if __name__=='__main__':main()
