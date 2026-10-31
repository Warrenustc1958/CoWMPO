# Copyright 2026 The RLinf Authors.
"""Replay frozen natural actions with RGB-estimated robot initialization."""
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
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
def install_robot(scene,record):
    r=scene.env.robots[0];sim=scene.env.sim
    for field,idx in [('arm_qpos',r._ref_joint_pos_indexes),('grip_qpos',r._ref_gripper_joint_pos_indexes)]:sim.data.qpos[idx]=record[field]
    for field,idx in [('arm_qvel',r._ref_joint_vel_indexes),('grip_qvel',r._ref_gripper_joint_vel_indexes)]:sim.data.qvel[idx]=record[field]
    sim.forward();r.gripper.current_action=np.asarray(record['grip_current_action']).copy()
    r.controller.update(force=True);r.controller.update_initial_joints(RESET_QPOS);r.controller.reset_goal()
def worker(key,robot_root,out):
    rack=key.startswith('h');probe_path=R/'rack_touch_dev_v1'/key/'probes/libero_goal_t0_i0/probe.json' if rack else R/'grasp_contrast_dev_v1'/key/'probe.json'
    p=json.loads(probe_path.read_text());ep=p['episode'];dest=out/key;dest.mkdir()
    scene=Scene(str(R/'stove_parts_geometry_dev_v1/world07'));scene.reset();apply_state(scene,json.loads((R/'stove_parts_states_dev_v1'/ep/'state.json').read_text())['objects'],require_matched=True)
    record=json.loads((robot_root/ep/'robot.json').read_text())['robot'];install_robot(scene,record)
    name='cream_cheese';k=scene.names.index(name);initial=scene.env.object_pos(name).copy();fingers=set();eef=[]
    for step,act in enumerate(p['actions'],1):
        scene.env.step(np.asarray(act));eef.append(scene.env.eef_pos())
        if step==p.get('phases',{}).get('holding'):
            held=scene.env.object_pos(name).copy();sim=scene.env.sim
            for c in sim.data.contact[:sim.data.ncon]:
                for g,other in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                    if scene.geom2obj[g]==k and 'finger' in (sim.model.geom_id2name(other) or ''):fingers.add(int(sim.model.geom_bodyid[other]))
        if step in p['frames']:
            frame=dest/f'f{step:03d}';frame.mkdir();seg=scene.seg();np.savez_compressed(frame/'masks.npz',robot=scene.robot_mask(),cream_cheese=seg==k)
    report={'key':key,'initial_robot_source':str(robot_root/ep/'robot.json')}
    if not rack:report['prediction']=bool(held[2]-initial[2]>.05 and len(fingers)>=2)
    scene.env.close();np.save(dest/'eef.npy',np.asarray(eef));(dest/'result.json').write_text(json.dumps(report));print('rgb_robot_replay',key,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--key');p.add_argument('--robot-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.key:return worker(a.key,a.robot_root,a.out)
    a.out.mkdir(exist_ok=False);keys=[f'libero_goal_t{k}_i0__{m}' for k in range(6) for m in ['grasp','miss']]+['h070','h170','h270']
    def run(k):
        with (a.out/(k+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',k,'--robot-root',str(a.robot_root),'--out',str(a.out)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((k,r.returncode))
    with ThreadPoolExecutor(8) as pool:list(pool.map(run,keys))
    print('RGB_ROBOT_REPLAY_DONE',flush=True)
if __name__=='__main__':main()
