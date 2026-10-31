# Copyright 2026 The RLinf Authors.
"""Freeze gripper-specific contact probes before Target execution."""
import argparse, hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha, verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'rack_vertical_preflight_dev_v3';EP='libero_goal_t0_i0'
BASE=R/'bowl_support_dev_v1/world11';STATE=R/'natural_states_all_dev_v1'/EP/'state.json'
ROBOT=R/'robot_rgb_all_dev_v2'/EP/'robot.json'
ALT=R/'rack_supported_silhouette_dev_v1/candidate01/world'
ALTSTATE=R/'rack_supported_response_dev_v2/candidate01'/EP/'state.json'
SPECS={f'vert_x{dx:+04d}_y{dy:+04d}':(dx,dy) for dx in [-80,0,80] for dy in [-80,0,80]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(key,variant):
    roles.require([EP],{'dev'},'isolated rack preflight')
    world,source=(BASE,STATE) if variant=='base' else (ALT,ALTSTATE)
    dest=OUT/key/variant;dest.mkdir(parents=True)
    scene=Scene(str(world));scene.reset();doc=json.loads(source.read_text())
    apply_state(scene,doc['objects'],require_matched=True,robot=json.loads(ROBOT.read_text())['robot'])
    m=scene.env.sim.model;gripper=np.array([(m.body_id2name(int(m.geom_bodyid[g])) or '').startswith('gripper0') for g in range(m.ngeom)])
    acts=[];rows=[];phases={};top=scene.world['table']['height']
    def advance(action,phase):
        scene.env.step(np.asarray(action));acts.append(list(map(float,action)));sim=scene.env.sim
        contacts=set();grip=set();arm=set()
        for c in sim.data.contact[:sim.data.ncon]:
            for rg,og in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                if not scene.robot_geom[rg] or scene.robot_geom[og]:continue
                obj=int(scene.geom2obj[og]);name=scene.names[obj] if obj>=0 else 'environment'
                contacts.add(name);(grip if gripper[rg] else arm).add(name)
        rows.append({'step':len(acts),'phase':phase,'eef':scene.env.eef_pos().tolist(),'contacts':sorted(contacts),
                     'gripper_contacts':sorted(grip),'arm_contacts':sorted(arm),
                     'max_arm_speed':float(np.max(np.abs(sim.data.qvel[scene.env.robots[0]._ref_joint_vel_indexes])))})
    if variant=='base':
        dx,dy=SPECS[key]
        from cowm.init.aabb import object_aabb
        center,_=object_aabb(m,scene.env.sim.data,np.nonzero(scene.geom2obj==scene.names.index('rack'))[0])
        start=center[:2]+np.array([dx,dy])/1000
        def drive(target,n,phase):
            for _ in range(n):
                a=np.zeros(7);a[:3]=np.clip(8*(np.asarray(target)-scene.env.eef_pos()),-.65,.65);a[6]=1;advance(a,phase)
            phases[phase]=len(acts)
        drive([*scene.env.eef_pos()[:2],top+.60],80,'raise')
        drive([*start,top+.60],100,'approach')
        drive([*start,top+.08],160,'contact')
        drive([*scene.env.eef_pos()[:2],top+.60],80,'retract')
        probe={'episode':EP,'obj':'rack','mode':'isolated_contact','key':key,'actions':acts,'frames':list(range(0,len(acts)+1,10)),
               'phases':phases,'world_tree_sha256':tree_sha(BASE),'state_sha256':sha(STATE),'robot_record_sha256':sha(ROBOT),
               'purpose':'adaptive dev design; vertical grid around Builder rack AABB center; adaptive after horizontal design; no Target coordinate or geometry input'}
        (OUT/key/'probe.json').write_text(json.dumps(probe,indent=1))
    else:
        probe=json.loads((OUT/key/'probe.json').read_text());phases=probe['phases']
        for step,act in enumerate(probe['actions'],1):advance(act,next(p for p,last in phases.items() if step<=last))
    scene.env.close()
    report={'key':key,'variant':variant,'rows':rows,
        'precontact_steps':sum(bool(r['contacts']) for r in rows if r['phase'] in ['raise','approach','descend']),
        'gripper_rack_steps':sum('rack' in r['gripper_contacts'] for r in rows if r['phase']=='contact'),
        'arm_contact_steps':sum(bool(r['arm_contacts']) for r in rows),
        'other_contact_steps':sum(bool(set(r['contacts'])-{'rack'}) for r in rows),
        'max_arm_speed':max(r['max_arm_speed'] for r in rows)}
    report['qualified']=report['precontact_steps']==report['arm_contact_steps']==report['other_contact_steps']==0 and report['max_arm_speed']<=2
    (dest/'result.json').write_text(json.dumps(report,indent=1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--key');p.add_argument('--variant');a=p.parse_args()
    if a.key:return worker(a.key,a.variant)
    OUT.mkdir(exist_ok=False);hashes={str(p):sha(p) for p in [STATE,ROBOT,ALTSTATE]};worlds={str(w):tree_sha(w) for w in [BASE,ALT]}
    (OUT/'freeze.json').write_text(json.dumps({'episode':EP,'grid':SPECS,'inputs':hashes,'worlds':worlds,'script_sha256':sha(__file__),
        'selection':'both worlds zero pre-contact and zero non-gripper/other-object contact, max arm speed <=2rad/s; at least one world >=5 gripper rack contact steps; >=3mm disagreement; top four spatially separated by >=8cm',
        'Target_diagnostic':'all queried probes retained; any contamination invalidates pure-rack interpretation, never silently remove samples',
        'GT_inputs':False},indent=1))
    def run(key):
        for variant in ['base','alternative']:
            with (OUT/(key+'_'+variant+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',key,'--variant',variant],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError((key,variant,r.returncode))
        b=json.loads((OUT/key/'base/result.json').read_text());c=json.loads((OUT/key/'alternative/result.json').read_text())
        delta=[np.linalg.norm(np.asarray(r['eef'])-s['eef'])*1000 for r,s in zip(b['rows'],c['rows']) if r['phase']=='contact']
        row={'key':key,'disagreement_mm':float(np.mean(delta)),'qualified':b['qualified'] and c['qualified'] and max(b['gripper_rack_steps'],c['gripper_rack_steps'])>=5,
             'worlds':[{k:v for k,v in q.items() if k!='rows'} for q in [b,c]]}
        print('PREFLIGHT',key,row['qualified'],round(row['disagreement_mm'],2),flush=True);return row
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,SPECS))
    selected=[]
    for row in sorted(rows,key=lambda r:-r['disagreement_mm']):
        if not row['qualified'] or row['disagreement_mm']<3:continue
        if all(np.linalg.norm(np.asarray(SPECS[row['key']])-SPECS[k])>=80 for k in selected):selected.append(row['key'])
        if len(selected)==4:break
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('preflight input changed')
    for w,h in worlds.items():verify_tree(w,h)
    report={'queries':rows,'selected':selected,'Target_queries_executed':0}
    (OUT/'selection.json').write_text(json.dumps(report,indent=1));print('SELECTED',selected,flush=True)
if __name__=='__main__':main()
