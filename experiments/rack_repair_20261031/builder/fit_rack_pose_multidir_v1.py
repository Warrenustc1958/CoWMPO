# Copyright 2026 The RLinf Authors.
"""Fit a shared local rack pose correction using orthogonal public contact responses."""
import argparse,copy,hashlib,itertools,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene,iou
from cowm.init.aabb import object_aabb
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'rack_pose_multidir_dev_v1';EP='libero_goal_t0_i0';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
PLAN=R/'rack_multidir_preflight_dev_v1';VIDEO=R/'rack_multidir_video_cpu_v1';OFFSETS=list(itertools.product([-.06,0.,.06],repeat=2))
FAMILIES=[('original',R/'bowl_support_dev_v1/world11',R/'natural_states_all_dev_v1'),
 ('supported',R/'rack_supported_silhouette_dev_v1/candidate01/world',R/'rack_supported_response_dev_v2/candidate01'),
 ('crossed',R/'rack_xframe_supported_silhouette_dev_v2/candidate03/world',R/'rack_xframe_supported_silhouette_dev_v2/candidate03/states')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worker(index):
    name,world,states=FAMILIES[index//len(OFFSETS)];offset=OFFSETS[index%len(OFFSETS)];dest=OUT/f'candidate{index:02d}';dest.mkdir();scene=Scene(str(world));static=[];docs={};wh=tree_sha(world)
    for ep in EPS:
        doc=json.loads((states/ep/'state.json').read_text());rack=doc['objects']['rack'];yaw=rack['yaw'];c,s=np.cos(yaw),np.sin(yaw);delta=np.array([[c,-s],[s,c]])@offset;rack['pos'][:2]=(np.asarray(rack['pos'][:2])+delta).tolist()
        robot=json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot'];scene.reset();apply_state(scene,doc['objects'],require_matched=True,robot=robot);k=scene.names.index('rack')
        with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:target=a['rack']
        pred=scene.seg()==k;score=iou(pred,target);static.append(score);center,half=object_aabb(scene.env.sim.model,scene.env.sim.data,np.nonzero(scene.geom2obj==k)[0]);rack.update(aabb_center_rel_base=(center-scene.base).tolist(),aabb_half=half.tolist(),pos_rel_base=(np.asarray(rack['pos'][:2])-scene.base[:2]).tolist(),iou=score)
        doc.update(world_sha256=scene.world['world_sha256'],world_tree_sha256=wh,robot=robot,robot_required=True);docs[ep]=doc;p=dest/'states'/ep;p.mkdir(parents=True);(p/'state.json').write_text(json.dumps(doc,indent=1));np.savez_compressed(p/'render_masks.npz',rack=pred)
    responses=[];keys=json.loads((OUT/'freeze.json').read_text())['keys']
    for key in keys:
        scene.reset();apply_state(scene,docs[EP]['objects'],require_matched=True,robot=docs[EP]['robot']);probe=json.loads((PLAN/key/'probe.json').read_text());values=[]
        for step,act in enumerate(probe['actions'],1):
            scene.env.step(np.asarray(act))
            if not (190<=step<=300 and step%10==0):continue
            frame=VIDEO/key/f'{EP}__f{step:03d}';rec=json.loads((frame/'association.json').read_text())
            if rec['masks_sha256']!=sha(frame/'masks.npz') or rec['objects']['robot']['status']!='matched':raise ValueError('invalid response evidence')
            with np.load(frame/'masks.npz') as a:target=a['robot']
            values.append(iou(scene.robot_mask(),target))
        if len(values)!=12:raise ValueError('incomplete contact response')
        responses.append({'key':key,'mean_iou':float(np.mean(values)),'ious':values})
    scene.env.close();verify_tree(world,wh);report={'index':index,'family':name,'local_offset_m':offset,'world':str(world),'world_tree_sha256':wh,'states':str(dest/'states'),'static_iou':float(np.mean(static)),'robot_iou':float(np.mean([v for r in responses for v in r['ious']])),'responses':responses}
    report['loss']=1-report['static_iou']+3*(1-report['robot_iou']);(dest/'result.json').write_text(json.dumps(report,indent=1))
def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int);a=p.parse_args()
    if a.index is not None:return worker(a.index)
    OUT.mkdir(exist_ok=False);files={};worlds={};keys=json.loads((PLAN/'selection.json').read_text())['selected']
    for name,w,s in FAMILIES:
        worlds[str(w)]=tree_sha(w)
        for ep in EPS:
            for q in [s/ep/'state.json',R/'robot_rgb_all_dev_v2'/ep/'robot.json',R/'visual_dev_bound_v1/dev3'/ep/'masks.npz']:files[str(q)]=sha(q)
    for key in keys:
        q=PLAN/key/'probe.json';files[str(q)]=sha(q)
        for step in range(190,301,10):
            for n in ['masks.npz','association.json']:
                q=VIDEO/key/f'{EP}__f{step:03d}'/n;files[str(q)]=sha(q)
    frozen={'keys':keys,'worlds':worlds,'files':files,'offsets':OFFSETS,'script_sha256':sha(__file__),'GT_inputs':False,'objective':'mean six-dev static loss + 3 times all 48 robot response frame loss','scope':'shared object-local correction, no episode-specific branches; adaptive dev'}
    (OUT/'freeze.json').write_text(json.dumps(frozen,indent=1))
    def run(i):
        with (OUT/f'candidate{i:02d}.log').open('w') as log:r=subprocess.run([sys.executable,__file__,'--index',str(i)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((i,r.returncode))
        print('RACK_POSE_CANDIDATE_COMPLETE',i,flush=True);return json.loads((OUT/f'candidate{i:02d}/result.json').read_text())
    with ThreadPoolExecutor(12) as pool:rows=list(pool.map(run,range(len(FAMILIES)*len(OFFSETS))))
    for q,h in files.items():
        if sha(q)!=h:raise ValueError('evidence changed')
    for w,h in worlds.items():verify_tree(w,h)
    brief=lambda r:{k:v for k,v in r.items() if k!='responses'};best=min(rows,key=lambda r:r['loss']);report={'best':brief(best),'candidates':[brief(r) for r in rows]};(OUT/'selection.json').write_text(json.dumps(report,indent=1));print('RACK_POSE_SELECTED',json.dumps(brief(best)),flush=True)
if __name__=='__main__':main()
