# Copyright 2026 The RLinf Authors.
"""Target-side contact attribution for frozen dev action queries; summaries only."""
import argparse
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm import roles
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/contact_diagnosis_dev_v1')

def worker(key):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    if key.startswith('h'):path=R/'rack_touch_dev_v1'/key/'probes/libero_goal_t0_i0/probe.json'
    else:path=R/'grasp_contrast_dev_v1'/key/'probe.json'
    probe=json.loads(path.read_text());ep=probe['episode'];roles.require([ep],{'dev'},'contact attribution')
    init=Path('/data_cf/cowmpo/init/libero_goal_dev3');manifest=json.loads((init/'manifest.json').read_text());info=manifest['episodes'][ep+'.npy']
    suite=benchmark.get_benchmark_dict()[manifest['suite']]();task=suite.get_task(info['task_id'])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
    env.reset();sim=env.env.sim;m=sim.model;priv=np.load(init/info['privileged']);m.body_pos[:]=priv['body_pos'];m.body_quat[:]=priv['body_quat']
    env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']]);warm=np.zeros(7);warm[6]=-1
    for _ in range(manifest['num_steps_wait']):env.step(warm)
    def robot(g):
        n=m.geom_id2name(g) or '';return n.startswith(('robot0','gripper0'))
    hits=Counter();first={};phases={};eef=[]
    for step,act in enumerate(probe['actions'],1):
        env.step(np.asarray(act));seen=set()
        for c in sim.data.contact[:sim.data.ncon]:
            for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                if robot(r) and not robot(g):
                    root=int(m.body_rootid[m.geom_bodyid[g]]);name=m.body_id2name(root) or m.geom_id2name(g) or str(root)
                    # Include contacting robot part to distinguish wrist/body obstruction from fingers.
                    part='finger' if 'finger' in (m.geom_id2name(r) or '') else 'other_robot'
                    seen.add(name+'/'+part)
        for n in seen:hits[n]+=1;first.setdefault(n,step)
        if step in probe.get('phases',{}).values():phases[next(k for k,v in probe['phases'].items() if v==step)]={'contact_objects':sorted(seen)}
        eef.append(sim.data.site_xpos[env.env.robots[0].eef_site_id].tolist())
    env.close()
    # Privileged trajectory stays Target-only, never copied into the public evidence tree.
    np.save(T/(key+'_eef.npy'),np.asarray(eef))
    report={'key':key,'contact_steps':dict(hits),'first_contact':first,'phases':phases}
    (T/(key+'.json')).write_text(json.dumps(report,indent=1));print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--key');a=p.parse_args()
    if a.key:return worker(a.key)
    T.mkdir(exist_ok=False);keys=[f'libero_goal_t{k}_i0__grasp' for k in range(6)]+['h070','h170','h270']
    def run(k):
        with (T/(k+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',k],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((k,r.returncode))
    with ThreadPoolExecutor(6) as pool:list(pool.map(run,keys))
    rows=[json.loads((T/(k+'.json')).read_text()) for k in keys]
    (T/'summary.json').write_text(json.dumps(rows,indent=1));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
