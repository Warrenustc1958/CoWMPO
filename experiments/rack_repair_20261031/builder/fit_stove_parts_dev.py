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
from PIL import Image
from cowm import roles
from cowm.init.estimate_state import Scene, iou
from cowm.init.aabb import object_aabb
from cowm.world.templates import _part, _wrap, _save
from cowm.world.artifact import tree_sha

R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
BASE=R/'world_pair_v1';OUT=R/'stove_parts_geometry_dev_v1'
EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
roles.require(EPS,{'dev'},'stove functional-part geometry development')

def build(world,length_factor,knob_height,offset_factor):
    meta=json.loads((world/'world.json').read_text())
    ob=next(o for o in meta['objects'] if o['name']=='stove');a=ob['attrs']
    W,D,h=a['width'],a['depth'],a['height'];L=length_factor*D;y=offset_factor*W
    body=_part('box',[D/2,W/2,h/2],[0,0,h/2],'0.8 0.8 0.75 1',density=5000.,friction=.6)
    body+=_part('cylinder',[.3*min(W,D),.002],[0,0,h+.002],'0.2 0.2 0.2 1',density=0.)
    body+=_part('box',[L/2,.15*W,.012],[-D/2-L/2,y,h/2],'0.05 0.05 0.05 1',density=5000.,friction=.6)
    body+=(f'<body name="knob" pos="{-D/2-.7*L:.5f} {y:.5f} {h/2+.012:.5f}">'
           '<joint name="knob" type="hinge" axis="1 0 0" limited="true" range="0 1.5708" damping="0.05" armature="0.0005"/>'
           +_part('cylinder',[.012,knob_height/2],[0,0,knob_height/2],'0.05 0.05 0.05 1',density=1100.,friction=.6)+'</body>')
    top=h/2+.012+knob_height
    _save(world/'assets/stove','stove',_wrap('stove','',body,top,max(W,D+L)/2))
    ob['top_z']=top
    ob['attrs'].update(knob_housing_length=L,knob_height=knob_height,knob_offset_y=y)
    meta.pop('world_sha256');meta['experimental_geometry']={'model':'generic external housing + hinged knob',
        'source':'declared knob joint and dev RGB-derived complete masks; no Target geometry','development_episodes':EPS}
    meta['world_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    (world/'world.json').write_text(json.dumps(meta,indent=1))

def worker(world,dest):
    world=Path(world);dest=Path(dest);dest.mkdir(exist_ok=False)
    scene=Scene(str(world));scene.reset();k=scene.names.index('stove');records=[]
    for ep in EPS:
        scene.reset()
        src=R/'state_visual_dev_merged_v1/dev3'/ep
        state=json.loads((src/'state.json').read_text())['objects']
        for i,n in enumerate(scene.names):scene.set_pose(i,state[n]['pos'][:2],state[n]['yaw'])
        with np.load(R/'complete_dev_masks_v1'/ep/'masks.npz') as a:masks={n:a[n] for n in a.files}
        old=state['stove'];best=(-1,None,None)
        for yaw0 in [0.,math.pi/2,math.pi,-math.pi/2]:
            xy=list(old['pos'][:2]);yaw=yaw0
            scene.set_pose(k,xy,yaw);score=iou(scene.seg()==k,masks['stove'])
            for step,angle in zip([.04,.02,.01,.005],[.4,.2,.1,.05]):
                for repeat in range(30):
                    candidates=[]
                    for dx,dy,da in [(step,0,0),(-step,0,0),(0,step,0),(0,-step,0),(0,0,angle),(0,0,-angle)]:
                        p=[xy[0]+dx,xy[1]+dy];q=yaw+da
                        scene.set_pose(k,p,q);candidates.append((iou(scene.seg()==k,masks['stove']),p,q))
                    candidate=max(candidates,key=lambda c:c[0])
                    if candidate[0]<=score+1e-6:break
                    score,xy,yaw=candidate
                scene.set_pose(k,xy,yaw)
            if score>best[0]:best=(score,xy,yaw)
        score,xy,yaw=best;scene.set_pose(k,xy,yaw)
        seg=scene.seg();m,d=scene.env.sim.model,scene.env.sim.data
        c,h=object_aabb(m,d,np.nonzero(scene.geom2obj==k)[0])
        state['stove'].update(pos=[*xy,scene.world['objects'][k]['pos'][2]],yaw=yaw,
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
    (dest/'score.json').write_text(json.dumps({'stove_sam_iou':score,'episodes':EPS}))
    print('STOVE_PART_SCORE',score,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--world');p.add_argument('--state');a=p.parse_args()
    if a.world:return worker(a.world,a.state)
    OUT.mkdir(exist_ok=False);jobs=[]
    for index,(length,height,offset) in enumerate(itertools.product([.3,.6,.9],[.04,.08],[-.25,0.,.25])):
        world=OUT/f'world{index:02d}';shutil.copytree(BASE,world);build(world,length,height,offset)
        jobs.append((index,world,{'length_factor':length,'knob_height':height,'offset_factor':offset}))
    def run(job):
        i,w,params=job;state=OUT/f'state{i:02d}'
        with (OUT/f'job{i:02d}.log').open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--world',str(w),'--state',str(state)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((i,result.returncode))
        score=json.loads((state/'score.json').read_text())['stove_sam_iou']
        print('stove_candidate_complete',i,score,flush=True)
        return {'index':i,'world':str(w),'state':str(state),'parameters':params,'score':score}
    with ThreadPoolExecutor(6) as pool:records=list(pool.map(run,jobs))
    best=max(records,key=lambda c:c['score'])
    (OUT/'selection.json').write_text(json.dumps({'selection':'dev SAM mask IoU only','candidates':records,'best':best},indent=1))
    print('STOVE_PARTS_DONE',json.dumps(best),flush=True)
if __name__=='__main__':main()
