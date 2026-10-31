# Copyright 2026 The RLinf Authors.
"""Target-only shape/pose substitution; oracle assets never leave Target storage."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/grasp_oracle_shape_dev_v1')
LABELS=T.parent/'grasp_contrast_dev_holdout_v1'

def worker(key,mode):
    p=json.loads((R/'grasp_contrast_dev_holdout_v1'/key/'probe.json').read_text());ep=p['episode']
    roles.require([ep],{'dev'},'Target oracle shape diagnosis')
    data='dev3' if roles.parse(ep)[1] in [0,1,2] else 'fit3'
    gt=json.loads((Path('/data_cf/cowmpo/target_eval/s5_gt')/data/(ep+'.json')).read_text())['cream_cheese']
    scene=Scene(str(T/'oracle_world'));scene.reset()
    state=json.loads((R/'stove_parts_states_dev_v1'/ep/'state.json').read_text())['objects']
    apply_state(scene,state,require_matched=True)
    name='cream_cheese';k=scene.names.index(name)
    if mode=='shape_pose':
        scene.set_pose(k,np.asarray(gt['aabb_center_rel_base'][:2])+scene.base[:2],gt['yaw'])
    initial=scene.env.object_pos(name).copy();fingers=set()
    for step,act in enumerate(p['actions'],1):
        scene.env.step(np.asarray(act))
        if step==p['phases']['holding']:
            held=scene.env.object_pos(name).copy();sim=scene.env.sim
            for c in sim.data.contact[:sim.data.ncon]:
                for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                    if scene.geom2obj[g]==k and 'finger' in (sim.model.geom_id2name(r) or ''):fingers.add(int(sim.model.geom_bodyid[r]))
    scene.env.close()
    (T/mode/(key+'.json')).write_text(json.dumps({'grasped':bool(held[2]-initial[2]>.05 and len(fingers)>=2)}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--key');p.add_argument('--mode');a=p.parse_args()
    if a.key:return worker(a.key,a.mode)
    T.mkdir(exist_ok=False);world=T/'oracle_world';shutil.copytree(R/'stove_parts_geometry_dev_v1/world07',world)
    gt=json.loads(Path('/data_cf/cowmpo/target_eval/s5_gt/dev3/libero_goal_t0_i0.json').read_text())['cream_cheese']
    angle=gt['yaw'];c,s=abs(np.cos(angle)),abs(np.sin(angle))
    xy=np.linalg.solve([[c,s],[s,c]],2*np.asarray(gt['aabb_half'][:2]))
    dims=[*xy,2*gt['aabb_half'][2]]
    if min(dims)<=0 or max(dims)>.3:raise ValueError('invalid oracle box reconstruction')
    meta=json.loads((world/'world.json').read_text());ob=next(o for o in meta['objects'] if o['name']=='cream_cheese')
    old=np.array([ob['attrs'][n] for n in ['size_x','size_y','size_z']]);scale=np.asarray(dims)/old
    tree=ET.parse(world/ob['xml'])
    for g in tree.iter('geom'):
        if g.get('type')!='box':raise ValueError('requires box')
        for attr in ['size','pos']:g.set(attr,' '.join(map(str,np.array(list(map(float,g.get(attr).split())))*scale)))
    tree.write(world/ob['xml']);ob['top_z']=dims[2]
    ob['attrs'].update(zip(['size_x','size_y','size_z'],dims));meta.pop('world_sha256')
    meta['target_only_diagnostic']='oracle shape; never Builder-visible'
    meta['world_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    (world/'world.json').write_text(json.dumps(meta,indent=1))
    keys=sorted(q.name for q in (R/'grasp_contrast_dev_holdout_v1').iterdir() if (q/'probe.json').exists())
    if len(keys)!=52:raise ValueError('expected52')
    reports={}
    for mode in ['shape','shape_pose']:
        (T/mode).mkdir()
        def run(key):
            with (T/mode/(key+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',key,'--mode',mode],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError((key,r.returncode))
        with ThreadPoolExecutor(12) as pool:list(pool.map(run,keys))
        rows=[(json.loads((LABELS/(k+'.json')).read_text())['label'],json.loads((T/mode/(k+'.json')).read_text())['grasped']) for k in keys]
        pos=sum(y for y,p in rows);neg=len(rows)-pos;tp=sum(y and p for y,p in rows);tn=sum(not y and not p for y,p in rows)
        reports[mode]={'TP':tp,'TN':tn,'FP':neg-tn,'FN':pos-tp,'balanced_accuracy':(tp/pos+tn/neg)/2}
        print('ORACLE',mode,reports[mode],flush=True)
    (T/'summary.json').write_text(json.dumps({'queries':52,'reports':reports,'scope':'oracle diagnostic only; no dimensions or poses exported'},indent=1))
if __name__=='__main__':main()
