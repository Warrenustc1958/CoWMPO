# Copyright 2026 The RLinf Authors.
"""Execute frozen rack-touch actions; RGB public, segmentation diagnosis Target-only."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from cowm import roles
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/rack_touch_dev_v1')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/rack_touch_dev_v1')
E=Path('/data_cf/cowmpo/evidence/codex_rack_touch_dev_v1');EP='libero_goal_t0_i0'
INIT=Path('/data_cf/cowmpo/init/libero_goal_dev3')
def worker(key):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    roles.require([EP],{'dev'},'Target rack touch')
    source=R/key/'probes'/EP/'probe.json';probe=json.loads(source.read_text())
    manifest=json.loads((INIT/'manifest.json').read_text());info=manifest['episodes'][EP+'.npy']
    suite=benchmark.get_benchmark_dict()[manifest['suite']]();task=suite.get_task(info['task_id'])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
        camera_heights=256,camera_widths=256,ignore_done=True)
    env.reset();sim=env.env.sim;model=sim.model;priv=np.load(INIT/info['privileged'])
    model.body_pos[:]=priv['body_pos'];model.body_quat[:]=priv['body_quat']
    obs=env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']])
    warm=np.zeros(7);warm[6]=-1
    for _ in range(manifest['num_steps_wait']):obs,_,_,_=env.step(warm)
    rroot=model.body_name2id(env.env.robots[0].robot_model.root_body)
    robot=np.array([int(model.body_rootid[model.geom_bodyid[g]])==rroot or
        (model.body_id2name(int(model.geom_bodyid[g])) or '').startswith(('robot0','gripper0')) for g in range(model.ngeom)])
    dest=E/key;dest.mkdir();diag=T/key;diag.mkdir();frames=set(probe['frames'])
    def save(step,obs):
        frame=f'{EP}__f{step:03d}';public=dest/frame;public.mkdir();private=diag/frame;private.mkdir()
        Image.fromarray(np.ascontiguousarray(obs['agentview_image'][::-1,::-1])).save(public/'frame0.png')
        seg=sim.render(width=256,height=256,camera_name='agentview',segmentation=True)
        gid=seg[...,1].astype(int);gid[(seg[...,0]!=5)|(gid<0)]=-1;gid=gid[::-1,::-1]
        mk=(gid>=0)&robot[np.clip(gid,0,None)]
        np.savez_compressed(private/'gt_masks.npz',robot=mk)
    if 0 in frames:save(0,obs)
    for step,act in enumerate(probe['actions'],1):
        obs,_,_,_=env.step(np.asarray(act))
        if step in frames:save(step,obs)
    env.close()
    record={'episode':EP,'height_key':key,'frames':probe['frames'],'query_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (dest/'query.json').write_text(json.dumps(record,indent=1));(diag/'query.json').write_text(json.dumps(record,indent=1))
def main():
    p=argparse.ArgumentParser();p.add_argument('--key');a=p.parse_args()
    if a.key:return worker(a.key)
    keys=['h070','h170','h270']
    if any(not (R/key/'probes'/EP/'probe.json').exists() for key in keys):raise ValueError('rack query planning incomplete')
    T.mkdir(exist_ok=False);E.mkdir(exist_ok=False)
    def run(key):
        with (T/(key+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--key',key],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((key,result.returncode))
        print('rack_touch_target_complete',key,flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(run,keys))
    print('RACK_TOUCH_TARGET_DONE',flush=True)
if __name__=='__main__':main()
