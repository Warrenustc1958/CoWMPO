# Copyright 2026 The RLinf Authors.
"""Refit one frozen geometry against bound masks while retaining other verified poses."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from cowm import roles
from cowm.init.estimate_state import Scene, iou
from cowm.init.observation import load_association, select_masks, state_issues
from cowm.init.aabb import object_aabb
from cowm.world.artifact import tree_sha, verify_tree

def main():
    p=argparse.ArgumentParser()
    for key in ['world','object','masks','states','out','role']:p.add_argument('--'+key,required=True)
    p.add_argument('--episode');p.add_argument('--jobs',type=int,default=6);a=p.parse_args()
    eps=sorted(q.name for q in Path(a.masks).iterdir() if (q/'association.json').is_file())
    roles.require(eps,{a.role},'frozen geometry pose validation')
    if a.episode:
        if a.episode not in eps:raise ValueError('unexpected episode')
        eps=[a.episode]
    out=Path(a.out)
    if not a.episode:
        out.mkdir(exist_ok=False);(out/'logs').mkdir()
        def run(ep):
            with (out/'logs'/(ep+'.log')).open('w') as log:
                args=[sys.executable,__file__]
                for k in ['world','object','masks','states','out','role']:args+=['--'+k,getattr(a,k)]
                result=subprocess.run(args+['--episode',ep],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError((ep,result.returncode))
            print('frozen_pose_complete',ep,flush=True)
        with ThreadPoolExecutor(a.jobs) as pool:list(pool.map(run,eps))
        print('FROZEN_POSES_DONE',len(eps),flush=True);return
    ep=a.episode;scene=Scene(a.world);scene.reset();world_hash=tree_sha(a.world)
    source=Path(a.states)/ep/'state.json';doc=json.loads(source.read_text());state=doc['objects']
    if state_issues(state,scene.names):raise ValueError('unverified input state')
    for i,n in enumerate(scene.names):scene.set_pose(i,state[n]['pos'][:2],state[n]['yaw'])
    root=Path(a.masks)/ep;association=load_association(root,ep)
    with np.load(root/'masks.npz') as arr:masks={n:arr[n] for n in arr.files}
    masks,status=select_masks(scene.names,masks,association)
    k=scene.names.index(a.object);mk=masks[a.object];old=state[a.object];best=(-1,None,None)
    for yaw0 in [0.,math.pi/2,math.pi,-math.pi/2]:
        xy=list(old['pos'][:2]);yaw=yaw0;scene.set_pose(k,xy,yaw);score=iou(scene.seg()==k,mk)
        for step,angle in zip([.04,.02,.01,.005],[.4,.2,.1,.05]):
            for repeat in range(30):
                candidates=[]
                for dx,dy,da in [(step,0,0),(-step,0,0),(0,step,0),(0,-step,0),(0,0,angle),(0,0,-angle)]:
                    pos=[xy[0]+dx,xy[1]+dy];theta=yaw+da;scene.set_pose(k,pos,theta)
                    candidates.append((iou(scene.seg()==k,mk),pos,theta))
                candidate=max(candidates,key=lambda c:c[0])
                if candidate[0]<=score+1e-6:break
                score,xy,yaw=candidate
            scene.set_pose(k,xy,yaw)
        if score>best[0]:best=score,xy,yaw
    score,xy,yaw=best;scene.set_pose(k,xy,yaw);seg=scene.seg()
    c,h=object_aabb(scene.env.sim.model,scene.env.sim.data,np.nonzero(scene.geom2obj==k)[0])
    state[a.object].update(pos=[*xy,scene.world['objects'][k]['pos'][2]],yaw=yaw,
       pos_rel_base=(np.array(xy)-scene.base[:2]).tolist(),aabb_center_rel_base=(c-scene.base).tolist(),aabb_half=h.tolist())
    for i,n in enumerate(scene.names):state[n]['iou']=iou(seg==i,masks[n])
    dest=out/ep;dest.mkdir()
    doc.update(world_sha256=scene.world['world_sha256'],world_tree_sha256=world_hash,objects=state,
        provenance={'parent_state_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'association_sha256':hashlib.sha256((root/'association.json').read_bytes()).hexdigest(),
         'method':'one-object pose fit; other verified poses retained','object':a.object})
    (dest/'state.json').write_text(json.dumps(doc,indent=1))
    np.savez_compressed(dest/'render_masks.npz',**{n:seg==i for i,n in enumerate(scene.names)})
    np.savez_compressed(dest/'masks.npz',**masks)
    data='dev3' if roles.parse(ep)[1] in [0,1,2] else 'fit3'
    rgb=np.asarray(Image.open(Path('/data_cf/cowmpo/evidence')/('libero_goal_'+data)/ep/'frame0.png').convert('RGB'))
    Image.fromarray(np.concatenate([rgb,scene.rgb()],axis=1)).save(dest/'overlay.png')
    verify_tree(a.world,world_hash);scene.env.close()
if __name__=='__main__':main()
