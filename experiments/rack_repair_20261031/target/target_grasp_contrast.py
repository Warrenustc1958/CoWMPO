# Copyright 2026 The RLinf Authors.
"""Target-only grasp contrast execution: RGB public, physical labels isolated."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from cowm import roles
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/grasp_contrast_dev_v1')
E=Path('/data_cf/cowmpo/evidence/codex_grasp_contrast_dev_v1')
INIT=Path('/data_cf/cowmpo/init/libero_goal_dev3')

def worker(key):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    probe=json.loads((R/'grasp_contrast_dev_v1'/key/'probe.json').read_text());ep=probe['episode']
    roles.require([ep],{'dev'},'Target grasp contrast')
    manifest=json.loads((INIT/'manifest.json').read_text());info=manifest['episodes'][ep+'.npy']
    suite=benchmark.get_benchmark_dict()[manifest['suite']]();task=suite.get_task(info['task_id'])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
        camera_heights=256,camera_widths=256,ignore_done=True)
    env.reset();sim=env.env.sim;model=sim.model;priv=np.load(INIT/info['privileged'])
    model.body_pos[:]=priv['body_pos'];model.body_quat[:]=priv['body_quat']
    obs=env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']])
    warm=np.zeros(7);warm[6]=-1
    for _ in range(manifest['num_steps_wait']):obs,_,_,_=env.step(warm)
    root=next(b for b in range(model.nbody) if int(model.body_rootid[b])==b and
              (model.body_id2name(b) or '').startswith('cream_cheese_'))
    initial=sim.data.body_xpos[root].copy();held=None;fingers=set();frames=set(probe['frames'])
    dest=E/key;dest.mkdir()
    def save(step,obs):
        frame=dest/f'f{step:03d}';frame.mkdir()
        Image.fromarray(np.ascontiguousarray(obs['agentview_image'][::-1,::-1])).save(frame/'frame0.png')
    if 0 in frames:save(0,obs)
    for step,act in enumerate(probe['actions'],1):
        obs,_,_,_=env.step(np.asarray(act))
        if step in frames:save(step,obs)
        if step==probe['phases']['holding']:
            held=sim.data.body_xpos[root].copy()
            for c in sim.data.contact[:sim.data.ncon]:
                for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                    if int(model.body_rootid[model.geom_bodyid[g]])==root and 'finger' in (model.geom_id2name(r) or ''):
                        fingers.add(int(model.geom_bodyid[r]))
    if held is None:raise ValueError('missing holding phase')
    grasped=bool(held[2]-initial[2]>.05 and len(fingers)>=2)
    final=sim.data.body_xpos[root].copy();env.close()
    result={'key':key,'episode':ep,'mode':probe['mode'],'label':grasped,
       'prediction':probe['prediction']['grasped'],'lift_cm':float((held[2]-initial[2])*100),
       'finger_bodies':len(fingers),'final_height_delta_cm':float((final[2]-initial[2])*100),
       'frames':len(frames)}
    (T/(key+'.json')).write_text(json.dumps(result,indent=1))
    # Public metadata contains only the action query and frame IDs.
    (dest/'query.json').write_text(json.dumps({k:probe[k] for k in ['episode','obj','mode','actions','frames','phases']},indent=1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--key');a=p.parse_args()
    if a.key:return worker(a.key)
    keys=sorted(q.name for q in (R/'grasp_contrast_dev_v1').iterdir() if (q/'probe.json').exists())
    if len(keys)!=12:raise ValueError('expected twelve frozen action contrasts')
    T.mkdir(exist_ok=False);E.mkdir(exist_ok=False)
    def run(key):
        with (T/(key+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--key',key],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((key,result.returncode))
        print('target_grasp_executed',key,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(run,keys))
    rows=[json.loads((T/(key+'.json')).read_text()) for key in keys]
    tp=sum(r['label'] and r['prediction'] for r in rows);tn=sum(not r['label'] and not r['prediction'] for r in rows)
    pos=sum(r['label'] for r in rows);neg=len(rows)-pos
    summary={'queries':len(rows),'frames':sum(r['frames'] for r in rows),'positive':pos,'negative':neg,
        'TP':tp,'TN':tn,'FP':neg-tn,'FN':pos-tp,'balanced_accuracy':(tp/pos+tn/neg)/2 if pos and neg else None,
        'scope':'six dev episodes; diagnostic only, not held-out P1 validation'}
    (T/'summary.json').write_text(json.dumps(summary,indent=1));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
