# Copyright 2026 The RLinf Authors.
"""Score frozen rack hypotheses against every new orthogonal-contact video frame."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from cowm.init.estimate_state import Scene,iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SC=Path(__file__).parent;OUT=R/'rack_multidir_candidates_dev_v1';EP='libero_goal_t0_i0';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
PLAN=R/'rack_multidir_preflight_dev_v1';VIDEO=R/'rack_multidir_video_cpu_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spec(seed):
    if seed==6:return R/'bowl_support_dev_v1/world11',R/'natural_states_all_dev_v1'
    return R/'rack_supported_silhouette_dev_v1'/f'candidate{seed:02d}'/'world',R/'rack_supported_response_dev_v2'/f'candidate{seed:02d}'
def worker(seed):
    frozen=json.loads((OUT/'freeze.json').read_text());world,states=spec(seed);scene=Scene(str(world));robot=json.loads((R/'robot_rgb_all_dev_v2'/EP/'robot.json').read_text())['robot'];static=[];responses=[]
    for ep in EPS:
        scene.reset();doc=json.loads((states/ep/'state.json').read_text());apply_state(scene,doc['objects'],require_matched=True,robot=json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot'])
        with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:target=a['rack']
        static.append(iou(scene.seg()==scene.names.index('rack'),target))
    for key in frozen['keys']:
        scene.reset();apply_state(scene,json.loads((states/EP/'state.json').read_text())['objects'],require_matched=True,robot=robot)
        probe=json.loads((PLAN/key/'probe.json').read_text());series=[]
        for step,act in enumerate(probe['actions'],1):
            scene.env.step(np.asarray(act))
            if not (190<=step<=300 and step%10==0):continue
            frame=VIDEO/key/f'{EP}__f{step:03d}';rec=json.loads((frame/'association.json').read_text())
            if rec['masks_sha256']!=sha(frame/'masks.npz') or rec['objects']['robot']['status']!='matched':raise ValueError('unusable response frame')
            with np.load(frame/'masks.npz') as a:target=a['robot']
            series.append(iou(scene.robot_mask(),target))
        if len(series)!=12:raise ValueError('incomplete contact frame set')
        responses.append({'key':key,'mean_iou':float(np.mean(series)),'ious':series})
    scene.env.close();report={'seed':seed,'world':str(world),'states':str(states),'static_iou':float(np.mean(static)),
        'robot_iou':float(np.mean([v for r in responses for v in r['ious']])),'responses':responses}
    report['loss']=1-report['static_iou']+3*(1-report['robot_iou']);(OUT/f'candidate{seed:02d}.json').write_text(json.dumps(report,indent=1))
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);a=p.parse_args()
    if a.seed is not None:return worker(a.seed)
    OUT.mkdir(exist_ok=False);complete=json.loads((VIDEO/'complete.json').read_text());keys=json.loads((PLAN/'selection.json').read_text())['selected']
    if {r['key'] for r in complete}!=set(keys) or sum(r['frames'] for r in complete)!=144:raise ValueError('video incomplete')
    files={};worlds={}
    for seed in range(7):
        w,s=spec(seed);worlds[str(w)]=tree_sha(w)
        for ep in EPS:
            for q in [s/ep/'state.json',R/'robot_rgb_all_dev_v2'/ep/'robot.json',R/'visual_dev_bound_v1/dev3'/ep/'masks.npz']:files[str(q)]=sha(q)
    for key in keys:
        q=PLAN/key/'probe.json';files[str(q)]=sha(q)
        for step in range(190,301,10):
            for n in ['masks.npz','association.json']:
                q=VIDEO/key/f'{EP}__f{step:03d}'/n;files[str(q)]=sha(q)
    (OUT/'freeze.json').write_text(json.dumps({'keys':keys,'files':files,'worlds':worlds,'script_sha256':sha(__file__),'GT_inputs':False,
        'objective':'six-dev static rack mask IoU plus 3 times robot-response loss, all 48 contact frames retained; adaptive dev experiment'},indent=1))
    def run(seed):
        with (OUT/f'candidate{seed:02d}.log').open('w') as log:r=subprocess.run([sys.executable,__file__,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((seed,r.returncode))
        return json.loads((OUT/f'candidate{seed:02d}.json').read_text())
    with ThreadPoolExecutor(7) as pool:rows=list(pool.map(run,range(7)))
    for q,h in files.items():
        if sha(q)!=h:raise ValueError('frozen evidence changed')
    for w,h in worlds.items():verify_tree(w,h)
    best=min(rows,key=lambda r:r['loss']);brief=lambda r:{k:v for k,v in r.items() if k!='responses'}
    report={'best':brief(best),'candidates':[brief(r) for r in rows]};(OUT/'selection.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)
if __name__=='__main__':main()
