# Copyright 2026 The RLinf Authors.
"""Validate supported rack silhouette candidates by real rendering and frozen contact responses."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from cowm.init.estimate_state import Scene,iou
from cowm.init.aabb import object_aabb
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SRC=R/'rack_supported_silhouette_dev_v1';OUT=R/'rack_supported_response_dev_v2';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
def worker(seed):
    source=SRC/f'candidate{seed:02d}';world=source/'world';dest=OUT/f'candidate{seed:02d}';dest.mkdir();scene=Scene(str(world));scores=[];dynamic=[];wh=tree_sha(world)
    for ep in EPS:
        scene.reset();doc=json.loads((source/'states'/ep/'state.json').read_text());robot=json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot'];apply_state(scene,doc['objects'],require_matched=True,robot=robot);seg=scene.seg();k=scene.names.index('rack')
        with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:masks={n:a[n] for n in a.files}
        overlap=iou(seg==k,masks['rack']);scores.append(overlap);c,h=object_aabb(scene.env.sim.model,scene.env.sim.data,np.nonzero(scene.geom2obj==k)[0]);doc['objects']['rack'].update(iou=overlap,aabb_center_rel_base=(c-scene.base).tolist(),aabb_half=h.tolist(),pos_rel_base=(np.asarray(doc['objects']['rack']['pos'][:2])-scene.base[:2]).tolist());doc.update(world_sha256=scene.world['world_sha256'],world_tree_sha256=wh,robot=robot,robot_required=True)
        p=dest/ep;p.mkdir();(p/'state.json').write_text(json.dumps(doc,indent=1));np.savez_compressed(p/'render_masks.npz',rack=seg==k);np.savez_compressed(p/'masks.npz',rack=masks['rack'])
    ep=EPS[0];doc=json.loads((dest/ep/'state.json').read_text())
    for key in ['h070','h170','h270']:
        scene.reset();apply_state(scene,doc['objects'],require_matched=True,robot=doc['robot']);probe=json.loads((R/'rack_touch_dev_v1'/key/'probes'/ep/'probe.json').read_text())
        for step,act in enumerate(probe['actions'],1):
            scene.env.step(np.asarray(act))
            if 120<=step<=220 and step%10==0:
                path=R/'rack_touch_video_v1'/key/f'{ep}__f{step:03d}'
                if json.loads((path/'association.json').read_text())['objects']['robot']['status']!='matched':raise ValueError('unmatched response mask')
                with np.load(path/'masks.npz') as a:target=a['robot']
                dynamic.append(iou(scene.robot_mask(),target))
    scene.env.close();verify_tree(world,wh);report={'seed':seed,'world':str(world),'world_tree_sha256':wh,'static_iou':float(np.mean(scores)),'static_per_ep':scores,'robot_iou':float(np.mean(dynamic)),'loss':float(1-np.mean(scores)+3*(1-np.mean(dynamic))),'Target_input':False};(dest/'result.json').write_text(json.dumps(report,indent=1));print('RACK_SUPPORTED_RESPONSE',json.dumps(report),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);a=p.parse_args()
    if a.seed is not None:return worker(a.seed)
    OUT.mkdir(exist_ok=False);(OUT/'freeze.json').write_text(json.dumps({'candidates':6,'selection':'1 - six-dev static rack SAM IoU + 3*(1 - frozen robot response IoU)','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'GT_inputs':False},indent=1))
    def run(seed):
        with (OUT/f'seed{seed}.log').open('w') as log:r=subprocess.run([sys.executable,__file__,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((seed,r.returncode))
        print('RACK_SUPPORTED_RESPONSE_COMPLETE',seed,flush=True)
        return json.loads((OUT/f'candidate{seed:02d}/result.json').read_text())
    with ThreadPoolExecutor(6) as pool:rows=list(pool.map(run,range(6)))
    best=min(rows,key=lambda x:x['loss']);(OUT/'selection.json').write_text(json.dumps({'best':best,'candidates':rows},indent=1));print('RACK_SUPPORTED_SELECTED',json.dumps(best),flush=True)
if __name__=='__main__':main()
