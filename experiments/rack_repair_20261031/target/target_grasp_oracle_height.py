# Copyright 2026 The RLinf Authors.
"""Target-only oracle-XY diagnosis; no GT pose or geometry is exported to Builder."""
import argparse
import hashlib
import shutil
import xml.etree.ElementTree as ET
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/grasp_oracle_height_dev_v1')
LABELS=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/grasp_contrast_dev_holdout_v1')
def worker(key):
    probe=json.loads((R/'grasp_contrast_dev_holdout_v1'/key/'probe.json').read_text());ep=probe['episode']
    roles.require([ep],{'dev'},'Target oracle XY diagnosis')
    data='dev3' if roles.parse(ep)[1] in [0,1,2] else 'fit3'
    gt=json.loads((Path('/data_cf/cowmpo/target_eval/s5_gt')/data/(ep+'.json')).read_text())
    scene=Scene(str(T/'oracle_world'));scene.reset()
    state=json.loads((R/'stove_parts_states_dev_v1'/ep/'state.json').read_text())['objects']
    apply_state(scene,state,require_matched=True)
    name='cream_cheese';k=scene.names.index(name)
    initial=scene.env.object_pos(name).copy();held=None;fingers=set()
    for step,act in enumerate(probe['actions'],1):
        scene.env.step(np.asarray(act))
        if step==probe['phases']['holding']:
            held=scene.env.object_pos(name).copy();sim=scene.env.sim
            for c in sim.data.contact[:sim.data.ncon]:
                for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                    if scene.geom2obj[g]==k and 'finger' in (sim.model.geom_id2name(r) or ''):fingers.add(int(sim.model.geom_bodyid[r]))
    scene.env.close()
    result={'key':key,'grasped':bool(held[2]-initial[2]>.05 and len(fingers)>=2)}
    (T/(key+'.json')).write_text(json.dumps(result))
def main():
    p=argparse.ArgumentParser();p.add_argument('--key');a=p.parse_args()
    if a.key:return worker(a.key)
    T.mkdir(exist_ok=False)
    world=T/'oracle_world';shutil.copytree(R/'stove_parts_geometry_dev_v1/world07',world)
    gt=json.loads(Path('/data_cf/cowmpo/target_eval/s5_gt/dev3/libero_goal_t0_i0.json').read_text())
    new_height=2*gt['cream_cheese']['aabb_half'][2]
    meta=json.loads((world/'world.json').read_text());ob=next(o for o in meta['objects'] if o['name']=='cream_cheese')
    scale=new_height/ob['attrs']['size_z'];tree=ET.parse(world/ob['xml'])
    for g in tree.iter('geom'):
        if g.get('type')!='box':raise ValueError('height diagnostic requires box template')
        size=list(map(float,g.get('size').split()));pos=list(map(float,g.get('pos').split()))
        size[2]*=scale;pos[2]*=scale
        g.set('size',' '.join(map(str,size)));g.set('pos',' '.join(map(str,pos)))
    tree.write(world/ob['xml']);ob['top_z']=new_height;ob['attrs']['size_z']=new_height
    meta.pop('world_sha256');meta['target_only_diagnostic']='oracle height; never Builder-visible'
    meta['world_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    (world/'world.json').write_text(json.dumps(meta,indent=1))
    keys=sorted(q.name for q in (R/'grasp_contrast_dev_holdout_v1').iterdir() if (q/'probe.json').exists())
    if len(keys)!=52:raise ValueError('expected exact 52 queries')
    def run(key):
        with (T/(key+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--key',key],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((key,result.returncode))
    with ThreadPoolExecutor(12) as pool:list(pool.map(run,keys))
    rows=[]
    for key in keys:
        label=json.loads((LABELS/(key+'.json')).read_text())['label']
        pred=json.loads((T/(key+'.json')).read_text())['grasped'];rows.append((label,pred))
    pos=sum(y for y,p in rows);neg=len(rows)-pos;tp=sum(y and p for y,p in rows);tn=sum(not y and not p for y,p in rows)
    report={'queries':len(rows),'TP':tp,'TN':tn,'FP':neg-tn,'FN':pos-tp,
      'balanced_accuracy':(tp/pos+tn/neg)/2,'scope':'oracle object height only; all XY/yaw/other parameters unchanged; Target-only diagnostic, never deployable'}
    (T/'summary.json').write_text(json.dumps(report,indent=1));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
