# Copyright 2026 The RLinf Authors.
"""Fit a generic extended knob housing using six dev RGB-derived masks only."""
import argparse
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.ndimage import sobel,binary_dilation,binary_erosion
from PIL import Image
from cowm import roles
from cowm.init.estimate_state import Scene, iou
from cowm.init.aabb import object_aabb
from cowm.world.templates import _part, _wrap, _save
from cowm.world.artifact import tree_sha

R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
BASE=R/'world_pair_v1';OUT=R/'rack_xframe_rgb_geometry_dev_v1'
EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
roles.require(EPS,{'dev'},'rack functional-part geometry development')

def build(world,width_factor,depth_factor,height_factor):
    meta=json.loads((world/'world.json').read_text())
    ob=next(o for o in meta['objects'] if o['name']=='rack');a=ob['attrs']
    W,D,H=a['width']*width_factor,a['depth']*depth_factor,a['height']*height_factor
    angle=math.atan2(H,D);length=math.hypot(H,D)
    body='';rgba='0.55 0.4 0.25 1'
    # Two crossed inclined slatted planes, with side rails; a generic folding drying rack.
    for sign in [-1,1]:
        tilt=sign*angle
        for j in range(12):
            t=(-.5+(j+.5)/12)*length
            body+=_part('box',[length/12*.43,W/2,.003],
                        [t*math.cos(tilt),0,H/2-t*math.sin(tilt)],rgba,
                        euler=f'0 {tilt} 0',density=600.,friction=.7)
        for side in [-1,1]:
            body+=_part('box',[length/2,.007,.007],[0,side*(W/2+.005),H/2],
                        '0.3 0.15 0.08 1',euler=f'0 {tilt} 0',density=600.,friction=.7)
    _save(world/'assets/rack','rack',_wrap('rack','',body,H,max(W,D)/2))
    ob['top_z']=H;ob['attrs'].update(width=W,depth=D,height=H)
    meta.pop('world_sha256');meta['experimental_geometry']={
        'model':'generic crossed slatted A-frame',
        'source':'dev RGB and masks only; no Target geometry','development_episodes':EPS}
    meta['world_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    (world/'world.json').write_text(json.dumps(meta,indent=1))

def edge_hist(rgb,mask):
    gray=np.asarray(rgb,dtype=float).mean(axis=2)
    gx=sobel(gray,axis=1);gy=sobel(gray,axis=0)
    weight=np.hypot(gx,gy)*mask
    angle=np.arctan2(gy,gx)%np.pi
    hist=np.histogram(angle,bins=16,range=(0,np.pi),weights=weight)[0]
    hist=(hist+np.roll(hist,1)+np.roll(hist,-1))/3
    return hist/max(np.linalg.norm(hist),1e-9)

def worker(world,dest):
    world=Path(world);dest=Path(dest);dest.mkdir(exist_ok=False)
    scene=Scene(str(world));scene.reset();k=scene.names.index('rack');records=[]
    for ep in EPS:
        scene.reset()
        src=R/'state_visual_dev_merged_v1/dev3'/ep
        state=json.loads((src/'state.json').read_text())['objects']
        for i,n in enumerate(scene.names):scene.set_pose(i,state[n]['pos'][:2],state[n]['yaw'])
        with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:masks={n:a[n] for n in a.files}
        old=state['rack'];best=(-1,None,None)
        target=np.asarray(Image.open(Path('/data_cf/cowmpo/evidence/libero_goal_dev3')/ep/'frame0.png').convert('RGB'))
        target_hist=edge_hist(target,binary_erosion(masks['rack'],iterations=1))
        def evidence_score():
            seg=scene.seg();mk=seg==k
            hist=edge_hist(scene.rgb(),binary_dilation(mk,iterations=1))
            return iou(mk,masks['rack'])+.25*float(hist@target_hist)
        for yaw0 in [0.,math.pi/2,math.pi,-math.pi/2]:
            xy=list(old['pos'][:2]);yaw=yaw0
            scene.set_pose(k,xy,yaw);score=evidence_score()
            for step,angle in zip([.04,.02,.01,.005],[.4,.2,.1,.05]):
                for repeat in range(30):
                    candidates=[]
                    for dx,dy,da in [(step,0,0),(-step,0,0),(0,step,0),(0,-step,0),(0,0,angle),(0,0,-angle)]:
                        p=[xy[0]+dx,xy[1]+dy];q=yaw+da
                        scene.set_pose(k,p,q);candidates.append((evidence_score(),p,q))
                    candidate=max(candidates,key=lambda c:c[0])
                    if candidate[0]<=score+1e-6:break
                    score,xy,yaw=candidate
                scene.set_pose(k,xy,yaw)
            if score>best[0]:best=(score,xy,yaw)
        score,xy,yaw=best;scene.set_pose(k,xy,yaw)
        seg=scene.seg();m,d=scene.env.sim.model,scene.env.sim.data
        c,h=object_aabb(m,d,np.nonzero(scene.geom2obj==k)[0])
        state['rack'].update(pos=[*xy,scene.world['objects'][k]['pos'][2]],yaw=yaw,
            pos_rel_base=(np.array(xy)-scene.base[:2]).tolist(),aabb_center_rel_base=(c-scene.base).tolist(),aabb_half=h.tolist(),iou=score)
        target=dest/ep;target.mkdir()
        (target/'state.json').write_text(json.dumps({'episode':ep,'objects':state,'world_sha256':scene.world['world_sha256'],
            'world_tree_sha256':tree_sha(world)},indent=1))
        np.savez_compressed(target/'render_masks.npz',**{n:seg==i for i,n in enumerate(scene.names)})
        np.savez_compressed(target/'masks.npz',**masks)
        rgb=np.asarray(Image.open(Path('/data_cf/cowmpo/evidence/libero_goal_dev3')/ep/'frame0.png').convert('RGB'))
        Image.fromarray(np.concatenate([rgb,scene.rgb()],axis=1)).save(target/'overlay.png')
        records.append(score)
    scene.env.close();score=float(np.mean(records))
    (dest/'score.json').write_text(json.dumps({'rack_sam_iou':score,'episodes':EPS}))
    print('RACK_PART_SCORE',score,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--world');p.add_argument('--state');a=p.parse_args()
    if a.world:return worker(a.world,a.state)
    OUT.mkdir(exist_ok=False);jobs=[]
    for index,(length,height,offset) in enumerate(itertools.product([1.,1.5],[2.,3.,4.],[.8,1.,1.2])):
        world=OUT/f'world{index:02d}';shutil.copytree(BASE,world);build(world,length,height,offset)
        jobs.append((index,world,{'width_factor':length,'depth_factor':height,'height_factor':offset}))
    def run(job):
        i,w,params=job;state=OUT/f'state{i:02d}'
        with (OUT/f'job{i:02d}.log').open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--world',str(w),'--state',str(state)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((i,result.returncode))
        score=json.loads((state/'score.json').read_text())['rack_sam_iou']
        print('rack_candidate_complete',i,score,flush=True)
        return {'index':i,'world':str(w),'state':str(state),'parameters':params,'score':score}
    with ThreadPoolExecutor(6) as pool:records=list(pool.map(run,jobs))
    best=max(records,key=lambda c:c['score'])
    (OUT/'selection.json').write_text(json.dumps({'selection':'dev SAM mask IoU + 0.25 gradient-orientation histogram cosine','candidates':records,'best':best},indent=1))
    print('RACK_PARTS_DONE',json.dumps(best),flush=True)
if __name__=='__main__':main()
