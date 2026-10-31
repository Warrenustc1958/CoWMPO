# Copyright 2026 The RLinf Authors.
"""Target-only oracle robot initialization. No privileged input enters Builder."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.env import RESET_QPOS
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
OUT=T/'grasp_oracle_robot_dev_v1'
def worker(key):
    p=json.loads((R/'grasp_contrast_dev_holdout_v1'/key/'probe.json').read_text());ep=p['episode']
    scene=Scene(str(R/'stove_parts_geometry_dev_v1/world07'));scene.reset();apply_state(scene,json.loads((R/'stove_parts_states_dev_v1'/ep/'state.json').read_text())['objects'],require_matched=True)
    with np.load(T/'robot_initial_dev_v1'/(ep+'.npz')) as a:s={k:a[k] for k in a.files}
    r=scene.env.robots[0];sim=scene.env.sim
    sim.data.qpos[r._ref_joint_pos_indexes]=s['arm_q'];sim.data.qpos[r._ref_gripper_joint_pos_indexes]=s['grip_q']
    sim.data.qvel[r._ref_joint_vel_indexes]=s['arm_qvel'];sim.data.qvel[r._ref_gripper_joint_vel_indexes]=s['grip_qvel'];sim.forward()
    r.gripper.current_action=s['grip_current_action'].copy();r.controller.update(force=True);r.controller.update_initial_joints(RESET_QPOS);r.controller.reset_goal()
    initial_error=float(np.linalg.norm(scene.env.eef_pos()-s['eef']))
    name='cream_cheese';k=scene.names.index(name);initial=scene.env.object_pos(name).copy();fingers=set()
    for step,act in enumerate(p['actions'],1):
        scene.env.step(np.asarray(act))
        if step==p['phases']['holding']:
            held=scene.env.object_pos(name).copy()
            for c in sim.data.contact[:sim.data.ncon]:
                for g,other in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                    if scene.geom2obj[g]==k and 'finger' in (sim.model.geom_id2name(other) or ''):fingers.add(int(sim.model.geom_bodyid[other]))
    scene.env.close();(OUT/(key+'.json')).write_text(json.dumps({'grasped':bool(held[2]-initial[2]>.05 and len(fingers)>=2),'initial_eef_error_m':initial_error}))
def main():
    p=argparse.ArgumentParser();p.add_argument('--key');a=p.parse_args()
    if a.key:return worker(a.key)
    OUT.mkdir(exist_ok=False);keys=sorted(p.parent.name for p in (R/'grasp_contrast_dev_holdout_v1').glob('*/probe.json'))
    def run(k):
        with (OUT/(k+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',k],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((k,r.returncode))
    with ThreadPoolExecutor(12) as pool:list(pool.map(run,keys))
    rows=[(json.loads((T/'grasp_contrast_dev_holdout_v1'/(k+'.json')).read_text())['label'],json.loads((OUT/(k+'.json')).read_text())['grasped']) for k in keys]
    pos=sum(y for y,p in rows);neg=len(rows)-pos;tp=sum(y and p for y,p in rows);tn=sum(not y and not p for y,p in rows)
    report={'queries':len(rows),'TP':tp,'TN':tn,'FP':neg-tn,'FN':pos-tp,'balanced_accuracy':(tp/pos+tn/neg)/2,'max_initial_eef_error_m':max(json.loads((OUT/(k+'.json')).read_text())['initial_eef_error_m'] for k in keys),'scope':'Target-private oracle robot state; not a deployable solution'}
    (OUT/'summary.json').write_text(json.dumps(report,indent=1));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
