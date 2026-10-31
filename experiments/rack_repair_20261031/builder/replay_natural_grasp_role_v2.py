# Copyright 2026 The RLinf Authors.
"""Frozen natural grasp replay with explicit world, object and RGB-robot estimates."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
from replay_rgb_robot import install_robot
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(a):
    probe=json.loads((a.probes/a.key/'probe.json').read_text());ep=probe['episode'];roles.require([ep],{a.role},'frozen RGB initialization validation')
    scene=Scene(str(a.world));scene.reset();apply_state(scene,json.loads((a.states/ep/'state.json').read_text())['objects'],require_matched=True)
    install_robot(scene,json.loads((a.robot/ep/'robot.json').read_text())['robot'])
    name='cream_cheese';k=scene.names.index(name);initial=scene.env.object_pos(name).copy();fingers=set()
    for step,act in enumerate(probe['actions'],1):
        scene.env.step(np.asarray(act))
        if step==probe['phases']['holding']:
            held=scene.env.object_pos(name).copy();sim=scene.env.sim
            for c in sim.data.contact[:sim.data.ncon]:
                for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                    if scene.geom2obj[g]==k and 'finger' in (sim.model.geom_id2name(r) or ''):fingers.add(int(sim.model.geom_bodyid[r]))
    scene.env.close();(a.out/(a.key+'.json')).write_text(json.dumps({'key':a.key,'episode':ep,'prediction':bool(held[2]-initial[2]>.05 and len(fingers)>=2),'lift_cm':float((held[2]-initial[2])*100),'finger_bodies':len(fingers)}))
def main():
    p=argparse.ArgumentParser()
    for n in ['world','states','robot','probes','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--role',choices=['dev','fit'],required=True);p.add_argument('--expected',type=int,required=True);p.add_argument('--key');p.add_argument('--jobs',type=int,default=8);a=p.parse_args()
    if a.key:return worker(a)
    a.out.mkdir(exist_ok=False);keys=[];sources={}
    for path in sorted(a.probes.glob('*/probe.json')):
        ep=json.loads(path.read_text())['episode'];state=a.states/ep/'state.json';robot=a.robot/ep/'robot.json'
        if state.exists() and robot.exists():keys.append(path.parent.name);sources.update({str(p):sha(p) for p in [state,robot,path]})
    if len(keys)!=a.expected:raise ValueError(('wrong exact query count',len(keys),a.expected))
    world_hash=tree_sha(a.world);(a.out/'freeze.json').write_text(json.dumps({'world':str(a.world),'world_tree_sha256':world_hash,'keys':keys,'sources':sources,'script_sha256':sha(Path(__file__)),'Target_inputs':False},indent=1))
    def run(k):
        cmd=[sys.executable,__file__,'--key',k,'--role',a.role,'--expected',str(a.expected)]
        for n in ['world','states','robot','probes','out']:cmd+=['--'+n,str(getattr(a,n))]
        with (a.out/(k+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((k,r.returncode))
    with ThreadPoolExecutor(a.jobs) as pool:list(pool.map(run,keys))
    verify_tree(a.world,world_hash)
    for p,h in sources.items():
        if sha(Path(p))!=h:raise ValueError('input changed during replay')
    print('NATURAL_GRASP_REPLAY_COMPLETE',len(keys),flush=True)
if __name__=='__main__':main()
