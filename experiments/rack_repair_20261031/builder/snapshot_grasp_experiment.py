# Copyright 2026 The RLinf Authors.
"""Actual grasp/lift/release probe and RGB-inclusive branch restoration. CPU, no Target input."""
import json
import math
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import drive

OUT=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/snapshot_grasp_v1')
OUT.mkdir(exist_ok=False)
scene=Scene('/data_cf/cowmpo/worlds/s5_v4/shape/world_fit')
obj='cream_cheese';k=scene.names.index(obj);sim=scene.env.sim

def finger_contacts():
    bodies=set()
    for c in sim.data.contact[:sim.data.ncon]:
        for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
            if scene.geom2obj[g]==k and scene.robot_geom[r]:
                name=sim.model.geom_id2name(r) or ''
                if 'finger' in name:bodies.add(int(sim.model.geom_bodyid[r]))
    return sorted(bodies)

def trial(yaw,zoffset):
    scene.reset()
    for index,name in enumerate(scene.names):
        scene.set_pose(index,[.05,0.] if name==obj else [-1.2,.3+index*.3],yaw if name==obj else 0.)
    acts=[]
    target=np.array([.05,0.,.9+zoffset])
    drive(scene,target+[0,0,.13],-1.,acts,60)
    drive(scene,target,-1.,acts,70)
    for _ in range(30):
        act=np.array([0,0,0,0,0,0,1.]);scene.env.step(act);acts.append(act.tolist())
    close=scene.env.object_pos(obj).copy()
    drive(scene,target+[0,0,.14],1.,acts,80)
    lift=scene.env.object_pos(obj).copy();contacts=finger_contacts()
    return {'yaw':yaw,'zoffset':zoffset,'close_position':close.tolist(),'lift_position':lift.tolist(),
            'finger_bodies':contacts,'lift_cm':float((lift[2]-close[2])*100)},acts

def replay(actions):
    rows=[]
    for action in actions:
        obs,_,done,_=scene.env.step(np.asarray(action))
        rows.append((obs['agentview_image'].copy(),sim.data.qpos.copy(),sim.data.qvel.copy(),bool(done)))
    return rows

def branch_check(actions):
    snap=scene.env.snapshot();a=replay(actions)
    scene.env.restore(snap)
    rng=np.random.default_rng(31)
    replay([np.r_[rng.uniform(-.4,.4,6),-1.] for _ in range(10)])
    scene.env.restore(snap);b=replay(actions)
    return all(all(np.array_equal(x,y) for x,y in zip(ra,rb)) for ra,rb in zip(a,b))

attempts=[];found=None
for yaw in [0.,math.pi/2]:
    for offset in [.018,.028,.008]:
        record,actions=trial(yaw,offset);attempts.append(record)
        print('grasp_attempt',record,flush=True)
        if record['lift_cm']>5 and len(record['finger_bodies'])>=2:
            found=(record,actions);break
    if found:break
result={'attempts':attempts,'grasp_observed':bool(found)}
if found:
    record,actions=found
    result['hold']={'two_fingers':len(finger_contacts())>=2,
                    'rgb_qpos_qvel_done_equal':branch_check([[0,0,0,0,0,0,1.]]*15)}
    released=[]
    for _ in range(12):
        action=[0,0,0,0,0,0,-1.];scene.env.step(np.asarray(action));released.append(action)
    result['release']={'no_finger_contact':not finger_contacts(),
                       'height':float(scene.env.object_pos(obj)[2]),
                       'rgb_qpos_qvel_done_equal':branch_check([[0,0,0,0,0,0,-1.]]*15)}
    (OUT/'grasp_probe.json').write_text(json.dumps({'object':obj,'actions':actions,'release_actions':released,
       'test_layout':'all other objects parked; generic box at [0.05,0]','selected':record},indent=1))
scene.env.close()
result['passed']=bool(found and all(result['hold'].values()) and result['release']['no_finger_contact']
                      and result['release']['rgb_qpos_qvel_done_equal'])
(OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] else 1)
