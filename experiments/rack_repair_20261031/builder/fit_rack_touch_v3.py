# Copyright 2026 The RLinf Authors.
"""Fit rack hypotheses against frozen RGB-derived robot response and static masks."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm import roles
from replay_rgb_robot import install_robot
from cowm.init.estimate_state import Scene,iou
from cowm.init.aabb import object_aabb
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'rack_touch_fit_dev_v3';EP='libero_goal_t0_i0'
roles.require([EP],{'dev'},'rack touch geometry development')
BASESTATE=R/'stove_parts_states_dev_v1'/EP/'state.json'
FAMILIES=[
 ('original',R/'stove_parts_geometry_dev_v1/world07',BASESTATE),
 ('xframe09',R/'rack_xframe_geometry_dev_v1/world09',R/'rack_xframe_geometry_dev_v1/state09'/EP/'state.json'),
 ('xframe17',R/'rack_xframe_geometry_dev_v1/world17',R/'rack_xframe_geometry_dev_v1/state17'/EP/'state.json'),
 ('two_shelf04',R/'rack_geometry_dev_v1/world04',R/'rack_geometry_dev_v1/state04'/EP/'state.json')]
KEYS=['h070','h170','h270'];OFFSETS=[-.24,-.16,-.08,0.,.08,.16,.24]
def worker(index):
    family=FAMILIES[index//len(OFFSETS)];dx=OFFSETS[index%len(OFFSETS)];name,world,source=family
    dest=OUT/f'candidate{index:02d}';dest.mkdir();scene=Scene(str(world));scene.reset();world_hash=tree_sha(world)
    state=json.loads(BASESTATE.read_text())['objects'];rack=json.loads(source.read_text())['objects']['rack']
    state['rack']={**rack,'pos':[rack['pos'][0]+dx,*rack['pos'][1:]]}
    with np.load(R/'visual_dev_bound_v1/dev3'/EP/'masks.npz') as a:static=a['rack']
    scores=[];centroid_errors=[];k=scene.names.index('rack');static_score=None;poses=[]
    for key in KEYS:
        scene.reset();apply_state(scene,state,require_matched=True);install_robot(scene,json.loads((R/'robot_rgb_all_dev_v2'/EP/'robot.json').read_text())['robot'])
        if static_score is None:
            seg=scene.seg();static_score=iou(seg==k,static)
            m,d=scene.env.sim.model,scene.env.sim.data;c,h=object_aabb(m,d,np.nonzero(scene.geom2obj==k)[0])
            state['rack'].update(aabb_center_rel_base=(c-scene.base).tolist(),aabb_half=h.tolist(),
              pos_rel_base=(np.asarray(state['rack']['pos'][:2])-scene.base[:2]).tolist(),iou=static_score)
            np.savez_compressed(dest/'render_masks.npz',**{n:seg==j for j,n in enumerate(scene.names)})
            with np.load(R/'visual_dev_bound_v1/dev3'/EP/'masks.npz') as a:
                np.savez_compressed(dest/'masks.npz',**{n:a[n] for n in a.files})
        probe=json.loads((R/'rack_touch_dev_v1'/key/'probes'/EP/'probe.json').read_text())
        series=[]
        for step,act in enumerate(probe['actions'],1):
            scene.env.step(np.asarray(act))
            if 120<=step<=220 and step%10==0:
                frame=f'{EP}__f{step:03d}';path=R/'rack_touch_video_v1'/key/frame
                rec=json.loads((path/'association.json').read_text())
                if rec['objects']['robot']['status']!='matched':raise ValueError('unusable robot response evidence')
                with np.load(path/'masks.npz') as a:target=a['robot']
                pred=scene.robot_mask();overlap=iou(pred,target)
                yp,xp=np.nonzero(pred);yt,xt=np.nonzero(target)
                error=float(np.linalg.norm([xp.mean()-xt.mean(),yp.mean()-yt.mean()]))
                series.append({'step':step,'robot_iou':overlap,'robot_centroid_error_px':error})
                scores.append(overlap);centroid_errors.append(error)
        poses.append({'query':key,'frames':series})
    doc=json.loads(BASESTATE.read_text());doc.update(objects=state,world_sha256=scene.world['world_sha256'],world_tree_sha256=world_hash)
    (dest/'state.json').write_text(json.dumps(doc,indent=1))
    robot_iou=float(np.mean(scores));loss=1-static_score+3*(1-robot_iou)
    record={'index':index,'family':name,'world':str(world),'world_tree_sha256':world_hash,'dx':dx,
      'static_iou':static_score,'robot_iou':robot_iou,'robot_centroid_error_px':float(np.mean(centroid_errors)),
      'loss':loss,'queries':poses,'state':str(dest/'state.json')}
    (dest/'result.json').write_text(json.dumps(record,indent=1));verify_tree(world,world_hash);scene.env.close()
    print(json.dumps({k:v for k,v in record.items() if k!='queries'}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int);a=p.parse_args()
    if a.index is not None:return worker(a.index)
    OUT.mkdir(exist_ok=False)
    frozen={'episode':EP,'families':[(n,str(w),tree_sha(w)) for n,w,s in FAMILIES],'offsets_x':OFFSETS,
      'objective':'1 - static rack mask IoU + 3*(1 - mean robot IoU at steps120..220)',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'robot_initialization_sha256':hashlib.sha256((R/'robot_rgb_all_dev_v2'/EP/'robot.json').read_bytes()).hexdigest(),
      'evidence':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'rack_touch_video_v1').glob('*/*/masks.npz')},
      'GT_input':False}
    if len(frozen['evidence'])!=84:raise ValueError('expected exact 84 RGB-derived response masks')
    (OUT/'freeze.json').write_text(json.dumps(frozen,indent=1))
    def run(i):
        with (OUT/f'candidate{i:02d}.log').open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--index',str(i)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((i,result.returncode))
        record=json.loads((OUT/f'candidate{i:02d}/result.json').read_text());print('touch_candidate_complete',i,record['loss'],flush=True)
        return record
    with ThreadPoolExecutor(4) as pool:records=list(pool.map(run,range(len(FAMILIES)*len(OFFSETS))))
    best=min(records,key=lambda c:c['loss']);(OUT/'selection.json').write_text(json.dumps({'best':best,'candidates':records},indent=1))
    print('RACK_TOUCH_FIT_DONE',json.dumps({k:v for k,v in best.items() if k!='queries'}),flush=True)
if __name__=='__main__':main()
