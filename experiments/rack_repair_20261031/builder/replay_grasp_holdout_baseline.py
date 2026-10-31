# Copyright 2026 The RLinf Authors.
"""Reference template predictions for the exact frozen natural grasp queries."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.s6.params import apply_params
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'grasp_reference_dev_holdout_v1'
def worker(key):
    probe=json.loads((R/'grasp_contrast_dev_holdout_v1'/key/'probe.json').read_text());ep=probe['episode']
    scene=Scene(str(R/'s6_ready_v1/reference_world_pair'));scene.reset()
    state=json.loads((R/'s6_identification_v1/reference_states'/ep/'state.json').read_text())['objects']
    apply_state(scene,state,require_matched=True);apply_params(scene,{'ceramic.friction':.815})
    k=scene.names.index('cream_cheese');initial=scene.env.object_pos('cream_cheese').copy();held=None;fingers=set()
    for step,act in enumerate(probe['actions'],1):
        scene.env.step(np.asarray(act))
        if step==probe['phases']['holding']:
            held=scene.env.object_pos('cream_cheese').copy();sim=scene.env.sim
            for c in sim.data.contact[:sim.data.ncon]:
                for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                    if scene.geom2obj[g]==k and 'finger' in (sim.model.geom_id2name(r) or ''):fingers.add(int(sim.model.geom_bodyid[r]))
    scene.env.close()
    prediction={'key':key,'grasped':bool(held[2]-initial[2]>.05 and len(fingers)>=2),
                'lift_cm':float((held[2]-initial[2])*100),'finger_bodies':len(fingers)}
    (OUT/(key+'.json')).write_text(json.dumps(prediction,indent=1))
def main():
    p=argparse.ArgumentParser();p.add_argument('--key');a=p.parse_args()
    if a.key:return worker(a.key)
    OUT.mkdir(exist_ok=False)
    keys=sorted(q.name for q in (R/'grasp_contrast_dev_holdout_v1').iterdir() if (q/'probe.json').exists())
    if len(keys)!=52:raise ValueError('expected 52 exact queries')
    def run(key):
        with (OUT/(key+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--key',key],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((key,result.returncode))
        print('reference_grasp_replayed',key,flush=True)
    with ThreadPoolExecutor(12) as pool:list(pool.map(run,keys))
    print('REFERENCE_GRASP_DONE',flush=True)
if __name__=='__main__':main()
