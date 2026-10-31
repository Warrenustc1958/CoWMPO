# Copyright 2026 The RLinf Authors.
"""Paired low/high release controls; physical predictions stay separate from RGB evidence."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state,drive
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'release_controls_dev_v2'
WORLD=R/'bowl_support_dev_v1/world11';REF=R/'s6_ready_v1/reference_world_pair'
EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def classify_release(initial,held,fingers,release,trajectory,robot_contact,support):
    trajectory=np.asarray(trajectory);release=np.asarray(release)
    acquired=bool(held[2]-initial[2]>.05 and fingers>=2)
    drop=float(max(0,release[2]-trajectory[:,2].min()))
    drift=float(np.max(np.linalg.norm(trajectory[:,:2]-release[:2],axis=1)))
    tail_span=float(np.max(np.linalg.norm(trajectory[-20:]-trajectory[-1],axis=1)))
    detached=not any(robot_contact[-20:]);supported=sum(support[-20:])>=15
    status='no_carry' if not acquired else ('dropped' if drop>.05 else ('stable' if drop<=.03 and drift<=.03 and tail_span<=.002 and detached and supported else 'unresolved'))
    return {'status':status,'acquired':acquired,'drop_m':drop,'horizontal_drift_m':drift,'tail_span_m':tail_span,'detached':detached,'supported':supported}

def contacts(scene,k):
    sim=scene.env.sim;m=sim.model;finger=set();robot=False;support=False
    for c in sim.data.contact[:sim.data.ncon]:
        for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
            if scene.geom2obj[g]!=k:continue
            if scene.robot_geom[r]:
                robot=True
                if 'finger' in (m.geom_id2name(r) or ''):finger.add(int(m.geom_bodyid[r]))
            else:support=True
    return len(finger),robot,support

def replay(key,variant):
    ep,mode=key.split('__');roles.require([ep],{'dev'},'release controls')
    world=WORLD if variant=='shape' else REF
    source=R/('natural_states_all_dev_v1' if variant=='shape' else 's6_ready_v1/reference_states_dev')/ep/'state.json'
    robot=R/'robot_rgb_all_dev_v2'/ep/'robot.json'
    scene=Scene(str(world));scene.reset();state=json.loads(source.read_text())['objects']
    apply_state(scene,state,require_matched=True,robot=json.loads(robot.read_text())['robot'])
    k=scene.names.index('cream_cheese');initial=scene.env.object_pos('cream_cheese').copy();dest=OUT/key
    if variant=='shape':
        dest.mkdir();old=json.loads((R/'grasp_contrast_dev_v1'/(ep+'__grasp')/'probe.json').read_text())
        acts=old['actions'][:old['phases']['holding']]
        for act in acts:scene.env.step(np.asarray(act))
        held=scene.env.object_pos('cream_cheese').copy();fingers=contacts(scene,k)[0]
        phases={'holding':len(acts)}
        if mode=='low':
            # The destination comes from the Builder object's original support pose.
            h=next(o for o in scene.world['objects'] if o['name']=='cream_cheese')['attrs']['size_z']
            target=np.array([*state['cream_cheese']['pos'][:2],scene.world['table']['height']+.6*h])
            drive(scene,target,1.,acts,90)
        else:
            for _ in range(90):
                act=[0,0,0,0,0,0,1.];scene.env.step(np.asarray(act));acts.append(act)
        phases['before_release']=len(acts);release=scene.env.object_pos('cream_cheese').copy()
        Image.fromarray(scene.rgb()).save(dest/'before_release.png')
        trajectory=[];rc=[];sp=[]
        for _ in range(100):
            act=[0,0,0,0,0,0,-1.];scene.env.step(np.asarray(act));acts.append(act)
            trajectory.append(scene.env.object_pos('cream_cheese').copy());_,r,s=contacts(scene,k);rc.append(r);sp.append(s)
        phases['settled']=len(acts);Image.fromarray(scene.rgb()).save(dest/'settled.png')
        probe={'key':key,'episode':ep,'obj':'cream_cheese','mode':mode,'actions':acts,'phases':phases,'frames':list(range(0,len(acts)+1,10)),
               'source_grasp_sha256':sha(R/'grasp_contrast_dev_v1'/(ep+'__grasp')/'probe.json')}
        (dest/'probe.json').write_text(json.dumps(probe,indent=1))
    else:
        probe=json.loads((dest/'probe.json').read_text());trajectory=[];rc=[];sp=[]
        for step,act in enumerate(probe['actions'],1):
            scene.env.step(np.asarray(act))
            if step==probe['phases']['holding']:held=scene.env.object_pos('cream_cheese').copy();fingers=contacts(scene,k)[0]
            if step==probe['phases']['before_release']:release=scene.env.object_pos('cream_cheese').copy()
            if step>probe['phases']['before_release']:
                trajectory.append(scene.env.object_pos('cream_cheese').copy());_,r,s=contacts(scene,k);rc.append(r);sp.append(s)
    prediction=classify_release(initial,held,fingers,release,trajectory,rc,sp);scene.env.close()
    (dest/(variant+'.json')).write_text(json.dumps({'key':key,'episode':ep,'prediction':prediction,'query_sha256':sha(dest/'probe.json')},indent=1))
    print(key,variant,prediction['status'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--key');p.add_argument('--variant');a=p.parse_args()
    if a.key:return replay(a.key,a.variant)
    OUT.mkdir(exist_ok=False);keys=[ep+'__'+mode for ep in EPS for mode in ['low','high']]
    files={}
    for ep in EPS:
        for p in [R/'natural_states_all_dev_v1'/ep/'state.json',R/'s6_ready_v1/reference_states_dev'/ep/'state.json',R/'robot_rgb_all_dev_v2'/ep/'robot.json',R/'grasp_contrast_dev_v1'/(ep+'__grasp')/'probe.json']:files[str(p)]=sha(p)
    worlds={str(w):tree_sha(w) for w in [WORLD,REF]}
    (OUT/'freeze.json').write_text(json.dumps({'keys':keys,'files':files,'worlds':worlds,'script_sha256':sha(__file__),
        'protocol':'same frozen grasp prefix, then lower to original support pose or hold elevated, then open for 100 steps; dev adaptive coverage study, not independent P1 validation',
        'label_rule':'acquired at holding: >5cm lift and >=2 finger bodies; no_carry retained; drop >5cm; stable <=3cm fall/drift, <=2mm final20step span, detached and support>=15/20; otherwise unresolved',
        'scoring':'report every key and acquisition coverage; confusion over stable/drop/no_carry/unresolved; no-carry excluded only from explicitly conditional release BA denominator; no forced binary labels','GT_inputs':False},indent=1))
    def run(key):
        for variant in ['shape','reference']:
            with (OUT/(key+'_'+variant+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',key,'--variant',variant],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError((key,variant,r.returncode))
        print('RELEASE_PLANNED',key,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(run,keys))
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('input changed')
    for w,h in worlds.items():verify_tree(w,h)
    (OUT/'complete.json').write_text(json.dumps({'keys':keys,'query_hashes':{k:sha(OUT/k/'probe.json') for k in keys}},indent=1))
if __name__=='__main__':main()
