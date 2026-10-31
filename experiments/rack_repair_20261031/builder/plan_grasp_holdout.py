# Copyright 2026 The RLinf Authors.
"""Natural-layout grasp/miss action contrasts planned from Builder estimates only."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state,drive
from cowm.world.artifact import tree_sha
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'grasp_contrast_dev_holdout_v1';WORLD=R/'stove_parts_geometry_dev_v1/world07'
EPS=sorted(q.name for q in (R/'stove_parts_states_dev_v1').iterdir() if (q/'state.json').exists() and q.name not in {f'libero_goal_t{k}_i0' for k in range(6)})
if len(EPS)!=26:raise ValueError('expected exact 26 unseen-outcome dev episodes')
roles.require(EPS,{'dev'},'natural grasp contrast development')

def worker(ep):
    scene=Scene(str(WORLD));obj='cream_cheese';k=scene.names.index(obj)
    source=R/'stove_parts_states_dev_v1'/ep/'state.json';state=json.loads(source.read_text())['objects']
    attrs=scene.world['objects'][k]['attrs'];height=attrs.get('height',.03)
    dims=attrs.get('size',[.05,.035,.03]);width=attrs.get('width',dims[0]);depth=attrs.get('depth',dims[1])
    for mode,offset in [('grasp',0.),('miss',2*max(width,depth))]:
        scene.reset();apply_state(scene,state,require_matched=True)
        initial=scene.env.object_pos(obj).copy();acts=[];phases={};dest=OUT/(ep+'__'+mode);dest.mkdir()
        target=np.array([*state[obj]['pos'][:2],scene.world['table']['height']+.6*height]);target[1]+=offset
        drive(scene,target+[0,0,.13],-1.,acts,60);phases['above']=len(acts)
        drive(scene,target,-1.,acts,70);phases['descend']=len(acts)
        for _ in range(30):
            action=[0,0,0,0,0,0,1.];scene.env.step(np.asarray(action));acts.append(action)
        phases['closed']=len(acts)
        drive(scene,target+[0,0,.14],1.,acts,80);phases['holding']=len(acts)
        held=scene.env.object_pos(obj).copy();bodies=set();sim=scene.env.sim
        for c in sim.data.contact[:sim.data.ncon]:
            for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                if scene.geom2obj[g]==k and 'finger' in (sim.model.geom_id2name(r) or ''):bodies.add(int(sim.model.geom_bodyid[r]))
        grasped=bool(held[2]-initial[2]>.05 and len(bodies)>=2)
        Image.fromarray(scene.rgb()).save(dest/'holding.png')
        for _ in range(45):
            action=[0,0,0,0,0,0,-1.];scene.env.step(np.asarray(action));acts.append(action)
        phases['released']=len(acts)
        Image.fromarray(scene.rgb()).save(dest/'released.png')
        frames=sorted({0,*phases.values(),*range(0,len(acts)+1,10)})
        record={'episode':ep,'obj':obj,'mode':mode,'actions':acts,'frames':frames,'phases':phases,
          'offset_y':offset,'world_tree_sha256':tree_sha(WORLD),'state_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'prediction':{'grasped':grasped,'lift_cm':float((held[2]-initial[2])*100),'finger_bodies':len(bodies)}}
        (dest/'probe.json').write_text(json.dumps(record,indent=1))
        print(ep,mode,record['prediction'],flush=True)
    scene.env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--episode');a=p.parse_args()
    if a.episode:return worker(a.episode)
    OUT.mkdir(exist_ok=False)
    freeze={'episodes':EPS,'world_tree_sha256':tree_sha(WORLD),'protocol':'same frozen grasp/miss design as first six dev; no outcome tuning','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'initial_states':'known dev states; action outcomes not yet queried'}
    (OUT/'freeze.json').write_text(json.dumps(freeze,indent=1))
    def run(ep):
        with (OUT/(ep+'.log')).open('w') as stream:
            result=subprocess.run([sys.executable,__file__,'--episode',ep],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((ep,result.returncode))
        print('grasp_plan_complete',ep,flush=True)
    with ThreadPoolExecutor(12) as pool:list(pool.map(run,EPS))
    print('GRASP_CONTRAST_PLANNED',flush=True)
if __name__=='__main__':main()
