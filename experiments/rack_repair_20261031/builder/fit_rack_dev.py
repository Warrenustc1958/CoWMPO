# Copyright 2026 The RLinf Authors.
"""Dev-only evidence fit of a generic slatted-rack hypothesis; no Target assets or GT."""
import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from cowm import roles
from cowm.init import estimate_state as ES
from cowm.init.observation import load_association
from cowm.world.artifact import tree_sha
from experimental_rack import build

R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
BASE=Path('/data_cf/cowmpo/worlds/s5_v4/shape/world_fit')
EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
roles.require(EPS,{'dev'},'rack geometry development')

def worker(world,out):
    ES.CONTACT_W=0.
    scene=ES.Scene(world);scene.reset();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    records=[];world_hash=tree_sha(world)
    for ep in EPS:
        scene.reset();root=R/'visual_dev_bound_v1/dev3'/ep
        with np.load(root/'masks.npz') as a:masks={n:a[n] for n in a.files}
        state,score,renders,seg=ES.estimate(scene,masks,load_association(root,ep))
        dest=out/ep;dest.mkdir()
        (dest/'state.json').write_text(json.dumps({'episode':ep,'world_sha256':scene.world['world_sha256'],
          'world_tree_sha256':world_hash,'objects':state,'score':score,'renders':renders},indent=1))
        np.savez_compressed(dest/'render_masks.npz',**{n:seg==i for i,n in enumerate(scene.names)})
        shutil.copy2(root/'masks.npz',dest/'masks.npz')
        records.append({n:r['iou'] for n,r in state.items()})
    mean={n:float(np.mean([r[n] for r in records])) for n in scene.names}
    (out/'score.json').write_text(json.dumps(mean,indent=2));scene.env.close()
    print(json.dumps(mean))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-world');parser.add_argument('--worker-out')
    args=parser.parse_args()
    if args.worker_world:return worker(args.worker_world,args.worker_out)
    work=R/'rack_geometry_dev_v1';work.mkdir(exist_ok=False)
    parent=json.loads((BASE/'world.json').read_text());rack=next(o for o in parent['objects'] if o['name']=='rack')
    jobs=[]
    for index,(wscale,dscale,incline) in enumerate(itertools.product([1.,1.5],[1.,2.],[0.,.4,.8])):
        world=work/f'world{index:02d}';shutil.copytree(BASE,world)
        meta=json.loads((world/'world.json').read_text());ob=next(o for o in meta['objects'] if o['name']=='rack')
        attrs=dict(ob['attrs']);attrs.update(width=attrs['width']*wscale,depth=attrs['depth']*dscale,incline=incline)
        build('rack',world/'assets/rack',width=attrs['width'],depth=attrs['depth'],height=attrs['height'],
              incline=incline,density=attrs.get('density',600.),friction=attrs.get('friction',.7))
        ob['attrs']=attrs
        meta.pop('world_sha256');meta['experimental_geometry']={'parent_tree_sha256':tree_sha(BASE),
              'model':'generic two-shelf slatted rack; no Target geometry','development_episodes':EPS}
        meta['world_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        (world/'world.json').write_text(json.dumps(meta,indent=2))
        jobs.append((index,world,attrs))
    def run(job):
        index,world,attrs=job;out=work/f'state{index:02d}'
        with (work/f'job{index:02d}.log').open('w') as log:
            done=subprocess.run([sys.executable,__file__,'--worker-world',str(world),'--worker-out',str(out)],
                                stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError(f'rack candidate {index} failed ({done.returncode})')
        score=json.loads((out/'score.json').read_text())
        print('candidate_complete',index,round(score['rack'],4),flush=True)
        return {'index':index,'world':str(world),'state':str(out),'attrs':attrs,'score':score}
    with ThreadPoolExecutor(4) as pool:records=list(pool.map(run,jobs))
    best=max(records,key=lambda r:r['score']['rack'])
    (work/'selection.json').write_text(json.dumps({'development_only':True,'episodes':EPS,'selection_metric':'SAM rack IoU',
          'candidates':records,'best':best},indent=2))
    print('RACK_DONE',json.dumps(best))

if __name__=='__main__':main()
