# Copyright 2026 The RLinf Authors.
"""Preflight orthogonal rack contacts in Builder before spending Target queries."""
import argparse, hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'rack_multidir_preflight_dev_v1';EP='libero_goal_t0_i0'
BASE=R/'bowl_support_dev_v1/world11';STATE=R/'natural_states_all_dev_v1'/EP/'state.json'
ROBOT=R/'robot_rgb_all_dev_v2'/EP/'robot.json'
ALT=R/'rack_supported_silhouette_dev_v1/candidate01/world'
ALTSTATE=R/'rack_supported_response_dev_v2/candidate01'/EP/'state.json'
KEYS=[f'{axis}{sign}_h{height:03d}' for axis in ['x','y'] for sign in ['m','p'] for height in [100,220]]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(key,variant):
    roles.require([EP],{'dev'},'rack multidir preflight')
    world=BASE if variant=='base' else ALT;source=STATE if variant=='base' else ALTSTATE
    dest=OUT/key/variant;dest.mkdir(parents=True)
    scene=Scene(str(world));scene.reset();doc=json.loads(source.read_text())
    apply_state(scene,doc['objects'],require_matched=True,robot=json.loads(ROBOT.read_text())['robot'])
    acts=[];rows=[];phases={};rack=scene.names.index('rack');top=scene.world['table']['height']
    def advance(action,phase):
        scene.env.step(np.asarray(action));acts.append(list(map(float,action)));sim=scene.env.sim;contacts=set()
        for c in sim.data.contact[:sim.data.ncon]:
            for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                obj=int(scene.geom2obj[g])
                if obj>=0 and scene.robot_geom[r]:contacts.add(scene.names[obj])
        if len(acts)%10==0:
            step=len(acts);p=dest/f'{EP}__f{step:03d}';p.mkdir()
            Image.fromarray(scene.rgb()).save(p/'frame0.png');np.savez_compressed(p/'masks.npz',robot=scene.robot_mask())
        rows.append({'step':len(acts),'phase':phase,'eef':scene.env.eef_pos().tolist(),'contacts':sorted(contacts)})
    if variant=='base':
        xy=np.asarray(doc['objects']['rack']['pos'][:2]);axis=0 if key[0]=='x' else 1;sign=-1 if key[1]=='m' else 1;height=int(key[-3:])/1000
        start=xy.copy();start[axis]+=sign*.24;end=xy.copy();end[axis]-=sign*.07
        def drive(target,n,phase):
            for _ in range(n):
                a=np.zeros(7);a[:3]=np.clip(10*(np.asarray(target)-scene.env.eef_pos()),-1,1);a[6]=1;advance(a,phase)
            phases[phase]=len(acts)
        drive([*scene.env.eef_pos()[:2],top+.45],50,'raise')
        drive([*start,top+.45],75,'approach')
        drive([*start,top+height],65,'descend')
        drive([*end,top+height],110,'contact')
        drive([*scene.env.eef_pos()[:2],top+.45],50,'retract')
        probe={'episode':EP,'obj':'rack','mode':'multidir_contact','key':key,'actions':acts,'frames':list(range(0,len(acts)+1,10)),
               'phases':phases,'world_tree_sha256':tree_sha(BASE),'state_sha256':sha(STATE),'robot_record_sha256':sha(ROBOT),
               'purpose':'different approach axes/heights, public RGB response only; no Target feedback used'}
        (OUT/key/'probe.json').write_text(json.dumps(probe,indent=1))
    else:
        probe=json.loads((OUT/key/'probe.json').read_text());phases=probe['phases']
        for step,act in enumerate(probe['actions'],1):advance(act,next(p for p,last in phases.items() if step<=last))
    scene.env.close();report={'key':key,'variant':variant,'world_tree_sha256':tree_sha(world),'rows':rows,
        'rack_contact_steps':sum('rack' in r['contacts'] for r in rows),
        'other_contact_steps':sum(bool(set(r['contacts'])-{'rack'}) for r in rows),
        'other_contact_objects':sorted(set(n for r in rows for n in r['contacts'] if n!='rack'))}
    (dest/'result.json').write_text(json.dumps(report,indent=1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--key');p.add_argument('--variant');a=p.parse_args()
    if a.key:return worker(a.key,a.variant)
    OUT.mkdir(exist_ok=False);hashes={str(p):sha(p) for p in [STATE,ROBOT,ALTSTATE]};worlds={str(w):tree_sha(w) for w in [BASE,ALT]}
    (OUT/'freeze.json').write_text(json.dumps({'episode':EP,'keys':KEYS,'inputs':hashes,'worlds':worlds,'script_sha256':sha(__file__),
        'selection':'no contact with other objects in either hypothesis; rank contact-phase EEF disagreement; at most four Target queries','GT_inputs':False},indent=1))
    def run(key):
        for variant in ['base','alternative']:
            with (OUT/(key+'_'+variant+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',key,'--variant',variant],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError((key,variant,r.returncode))
        b=json.loads((OUT/key/'base/result.json').read_text());c=json.loads((OUT/key/'alternative/result.json').read_text())
        values=[np.linalg.norm(np.asarray(r['eef'])-s['eef'])*1000 for r,s in zip(b['rows'],c['rows']) if r['phase']=='contact']
        return {'key':key,'mean_contact_eef_disagreement_mm':float(np.mean(values)),'max_contact_eef_disagreement_mm':float(max(values)),
                'base_rack_contact_steps':b['rack_contact_steps'],'alt_rack_contact_steps':c['rack_contact_steps'],
                'other_contact_objects':sorted(set(b['other_contact_objects']+c['other_contact_objects'])),
                'safe':b['other_contact_steps']==c['other_contact_steps']==0}
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,KEYS))
    selected=sorted([r for r in rows if r['safe'] and r['mean_contact_eef_disagreement_mm']>=3],key=lambda r:-r['mean_contact_eef_disagreement_mm'])[:4]
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('preflight input changed')
    for w,h in worlds.items():verify_tree(w,h)
    report={'queries':rows,'selected':[r['key'] for r in selected],'Target_queries_executed':0}
    (OUT/'selection.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)
if __name__=='__main__':main()
