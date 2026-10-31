# Copyright 2026 The RLinf Authors.
"""Compare three existing rack families using every new isolated-contact query."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from cowm.init.estimate_state import Scene,iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'rack_isolated_candidates_dev_v3';EP='libero_goal_t0_i0';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
SETS=[('horizontal',R/'rack_isolated_preflight_dev_v2',R/'rack_isolated_video_cpu_v2'),('vertical',R/'rack_vertical_preflight_dev_v3',R/'rack_vertical_video_cpu_v3')]
SPECS={'original':(R/'bowl_support_dev_v1/world11',R/'natural_states_all_dev_v1'),
       'supported':(R/'rack_supported_silhouette_dev_v1/candidate01/world',R/'rack_supported_response_dev_v2/candidate01'),
       'xframe':(R/'rack_xframe_supported_silhouette_dev_v2/candidate05/world',R/'rack_xframe_supported_silhouette_dev_v2/candidate05/states')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def query_records():
    records=[]
    for group,plan,video in SETS:
        keys=json.loads((plan/'selection.json').read_text())['selected']
        for key in keys:
            source=plan/key/'probe.json';probe=json.loads(source.read_text());phases=list(probe['phases']);idx=phases.index('contact')
            start=probe['phases'][phases[idx-1]];end=probe['phases']['contact'];frames=[s for s in probe['frames'] if start<s<=end]
            records.append({'group':group,'key':key,'source':str(source),'video':str(video/key),'frames':frames})
    return records

def prepare():
    OUT.mkdir(exist_ok=False);records=query_records();files={};worlds={}
    for world,states in SPECS.values():
        worlds[str(world)]=tree_sha(world)
        for ep in EPS:
            for p in [states/ep/'state.json',R/'robot_rgb_all_dev_v2'/ep/'robot.json',R/'visual_dev_bound_v1/dev3'/ep/'masks.npz']:files[str(p)]=sha(p)
    for r in records:files[r['source']]=sha(r['source'])
    config={'queries':records,'candidates':{k:list(map(str,v)) for k,v in SPECS.items()},'files':files,'worlds':worlds,'script_sha256':sha(__file__),
        'objective':'1-mean six-dev static rack IoU + 3*(1-equal-query mean contact-phase robot mask IoU); all four new queries retained; no Target contact/GT labels in selection',
        'scope':'three pre-existing family representatives only; adaptive dev functional discrimination, not optimized geometry or held-out validation','GT_inputs':False}
    (OUT/'freeze.json').write_text(json.dumps(config,indent=1));print('RACK_COMPARE_FROZEN',len(records),sum(len(r['frames']) for r in records),flush=True)

def worker(name):
    cfg=json.loads((OUT/'freeze.json').read_text());world,states=map(Path,cfg['candidates'][name]);scene=Scene(str(world));static=[];responses=[]
    robot=json.loads((R/'robot_rgb_all_dev_v2'/EP/'robot.json').read_text())['robot']
    for ep in EPS:
        scene.reset();apply_state(scene,json.loads((states/ep/'state.json').read_text())['objects'],require_matched=True,robot=json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot'])
        with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:target=a['rack']
        static.append(iou(scene.seg()==scene.names.index('rack'),target))
    for rec in cfg['queries']:
        scene.reset();apply_state(scene,json.loads((states/EP/'state.json').read_text())['objects'],require_matched=True,robot=robot)
        probe=json.loads(Path(rec['source']).read_text());values=[]
        for step,act in enumerate(probe['actions'],1):
            scene.env.step(np.asarray(act))
            if step not in rec['frames']:continue
            frame=Path(rec['video'])/f'{EP}__f{step:03d}';association=json.loads((frame/'association.json').read_text())
            if association['masks_sha256']!=sha(frame/'masks.npz') or association['objects']['robot']['status']!='matched':raise ValueError('unusable evidence')
            with np.load(frame/'masks.npz') as a:target=a['robot']
            values.append(iou(scene.robot_mask(),target))
        if len(values)!=len(rec['frames']):raise ValueError('frame set incomplete')
        responses.append({'key':rec['key'],'group':rec['group'],'mean_iou':float(np.mean(values)),'ious':values})
    scene.env.close();response=float(np.mean([r['mean_iou'] for r in responses]));st=float(np.mean(static))
    row={'name':name,'static_iou':st,'robot_iou':response,'loss':1-st+3*(1-response),'responses':responses}
    (OUT/(name+'.json')).write_text(json.dumps(row,indent=1));print(name,st,response,row['loss'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--candidate');a=p.parse_args()
    if a.prepare:return prepare()
    if a.candidate:return worker(a.candidate)
    cfg=json.loads((OUT/'freeze.json').read_text())
    if cfg['script_sha256']!=sha(__file__):raise ValueError('evaluation code changed')
    files={}
    for group,plan,video in SETS:
        complete=json.loads((video/'complete.json').read_text());expected=json.loads((plan/'selection.json').read_text())['selected']
        if {r['key'] for r in complete}!=set(expected):raise ValueError('missing complete query')
    for r in cfg['queries']:
        for step in r['frames']:
            for name in ['masks.npz','association.json']:
                q=Path(r['video'])/f'{EP}__f{step:03d}'/name;files[str(q)]=sha(q)
    (OUT/'evidence_manifest.json').write_text(json.dumps(files,indent=1))
    def run(name):
        with (OUT/(name+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--candidate',name],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((name,r.returncode))
        return json.loads((OUT/(name+'.json')).read_text())
    with ThreadPoolExecutor(3) as pool:rows=list(pool.map(run,cfg['candidates']))
    for f,h in {**cfg['files'],**files}.items():
        if sha(f)!=h:raise ValueError('input changed')
    for w,h in cfg['worlds'].items():verify_tree(w,h)
    report={'best':min(rows,key=lambda r:r['loss'])['name'],'candidates':rows}
    (OUT/'selection.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)
if __name__=='__main__':main()
