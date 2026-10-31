# Copyright 2026 The RLinf Authors.
"""Private factorial initial-state attribution for two exposed release counterexamples."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.init.aabb import object_aabb
from cowm.init.robot_state import install_robot_state
from cowm.s6.plan_probe import apply_state
from cowm.s6.params import apply_params
from plan_release_dev_v2 import contacts
from release_events_v3 import classify_event
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');OUT=T/'release_initial_attribution_v1'
KEYS=['libero_goal_t3_i0__high','libero_goal_t4_i0__high'];VARIANTS=['baseline','robot','object_pose','both']

def worker(key,variant):
    probe=json.loads((R/'release_controls_dev_v2'/key/'probe.json').read_text());ep=probe['episode'];world=R/'bowl_support_dev_v1/world11';scene=Scene(str(world));scene.reset();doc=json.loads((R/'natural_states_all_dev_v1'/ep/'state.json').read_text())
    apply_state(scene,doc['objects'],require_matched=True,robot=json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot'])
    apply_params(scene,json.loads((R/'s6_natural_v3/fit/s6_report.json').read_text())['variants']['param+struct']['params'])
    if variant in ['robot','both']:
        with np.load(T/'robot_initial_dev_v1'/(ep+'.npz')) as a:
            robot={'arm_qpos':a['arm_q'].tolist(),'arm_qvel':a['arm_qvel'].tolist(),'grip_qpos':a['grip_q'].tolist(),'grip_qvel':a['grip_qvel'].tolist(),'grip_current_action':a['grip_current_action'].tolist()}
        install_robot_state(scene,robot)
    k=scene.names.index('cream_cheese')
    if variant in ['object_pose','both']:
        gt=json.loads((Path('/data_cf/cowmpo/target_eval/s5_gt/dev3')/(ep+'.json')).read_text())['cream_cheese']
        old=np.asarray(doc['objects']['cream_cheese']['pos']);scene.set_pose(k,old[:2],gt['yaw'],z=old[2]);c,_=object_aabb(scene.env.sim.model,scene.env.sim.data,np.nonzero(scene.geom2obj==k)[0]);new=old+np.asarray(gt['aabb_center_rel_base'])+scene.base-c
        scene.set_pose(k,new[:2],gt['yaw'],z=new[2])
    initial=scene.env.object_pos('cream_cheese').copy();positions=[];rc=[];sp=[]
    for step,act in enumerate(probe['actions'],1):
        scene.env.step(np.asarray(act))
        if step==probe['phases']['holding']:held=scene.env.object_pos('cream_cheese').copy();fingers=contacts(scene,k)[0]
        if step>=probe['phases']['holding']:
            positions.append(scene.env.object_pos('cream_cheese').copy());_,r,s=contacts(scene,k);rc.append(r);sp.append(s)
    result=classify_event(initial,held,fingers,positions,rc,sp);scene.env.close()
    if result['event_offset'] is not None:result['event_step']=probe['phases']['holding']+result['event_offset']
    (OUT/(key+'_'+variant+'.json')).write_text(json.dumps({'key':key,'variant':variant,'result':result},indent=1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--key');p.add_argument('--variant');a=p.parse_args()
    if a.key:return worker(a.key,a.variant)
    OUT.mkdir(exist_ok=False)
    def run(pair):
        k,v=pair
        with (OUT/(k+'_'+v+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',k,'--variant',v],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((pair,r.returncode))
        return json.loads((OUT/(k+'_'+v+'.json')).read_text())
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,[(k,v) for k in KEYS for v in VARIANTS]))
    labels={k:json.loads((T/'release_event_dev_v3'/(k+'.json')).read_text())['label'] for k in KEYS}
    report={'scope':'Target-private two-dev diagnosis only; same geometry and actions; object pose replacement uses center/yaw only, not exact shape; no privileged state export','target':labels,'comparisons':rows}
    (OUT/'summary.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)
if __name__=='__main__':main()
