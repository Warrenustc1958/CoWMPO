# Copyright 2026 The RLinf Authors.
"""Re-score frozen release controls from physical separation, without changing actions."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
from plan_release_dev_v2 import contacts
from release_events_v3 import classify_event,self_test
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');PLAN=R/'release_controls_dev_v2';OUT=R/'release_event_dev_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(key,variant):
    probe=json.loads((PLAN/key/'probe.json').read_text());ep=probe['episode'];roles.require([ep],{'dev'},'release event diagnostic')
    world=R/('bowl_support_dev_v1/world11' if variant=='shape' else 's6_ready_v1/reference_world_pair')
    source=R/('natural_states_all_dev_v1' if variant=='shape' else 's6_ready_v1/reference_states_dev')/ep/'state.json'
    robot=R/'robot_rgb_all_dev_v2'/ep/'robot.json';scene=Scene(str(world));scene.reset()
    apply_state(scene,json.loads(source.read_text())['objects'],require_matched=True,robot=json.loads(robot.read_text())['robot'])
    k=scene.names.index('cream_cheese');initial=scene.env.object_pos('cream_cheese').copy();positions=[];rc=[];sp=[]
    for step,act in enumerate(probe['actions'],1):
        scene.env.step(np.asarray(act))
        if step==probe['phases']['holding']:held=scene.env.object_pos('cream_cheese').copy();fingers=contacts(scene,k)[0]
        if step>=probe['phases']['holding']:
            positions.append(scene.env.object_pos('cream_cheese').copy());_,rob,support=contacts(scene,k);rc.append(rob);sp.append(support)
    prediction=classify_event(initial,held,fingers,positions,rc,sp);scene.env.close()
    if prediction['event_offset'] is not None:prediction['event_step']=probe['phases']['holding']+prediction['event_offset']
    (OUT/key/(variant+'.json')).write_text(json.dumps({'key':key,'episode':ep,'prediction':prediction,'query_sha256':sha(PLAN/key/'probe.json')},indent=1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--key');p.add_argument('--variant');a=p.parse_args()
    if a.key:return worker(a.key,a.variant)
    OUT.mkdir(exist_ok=False);old=json.loads((PLAN/'freeze.json').read_text());complete=json.loads((PLAN/'complete.json').read_text());keys=complete['keys']
    for key in keys:
        if sha(PLAN/key/'probe.json')!=complete['query_hashes'][key]:raise ValueError('query changed')
        (OUT/key).mkdir()
    (OUT/'freeze.json').write_text(json.dumps({'keys':keys,'files':old['files'],'worlds':old['worlds'],'query_hashes':complete['query_hashes'],
        'script_sha256':sha(__file__),'rules_sha256':sha(Path(__file__).with_name('release_events_v3.py')),'tests':self_test(),
        'rule_change':'post hoc dev measurement correction: first >=10 steps no robot contact after acquired holding; measure fall after separation, including before scheduled opening; controlled lowering excluded; old results retained','GT_inputs':False},indent=1))
    def run(pair):
        key,variant=pair
        with (OUT/(key+'_'+variant+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',key,'--variant',variant],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((pair,r.returncode))
    with ThreadPoolExecutor(4) as pool:list(pool.map(run,[(key,v) for key in keys for v in ['shape','reference']]))
    for p,h in old['files'].items():
        if sha(p)!=h:raise ValueError('input changed')
    for w,h in old['worlds'].items():verify_tree(w,h)
    (OUT/'complete.json').write_text(json.dumps({'keys':keys},indent=1));print('RELEASE_EVENT_REPLAY_DONE',len(keys),flush=True)
if __name__=='__main__':main()
