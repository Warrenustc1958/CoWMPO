# Copyright 2026 The RLinf Authors.
"""Target-only dev replay/segmentation with exact saved RGB alignment checks."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from cowm import roles

S=Path('/data_cf/cowmpo/staging/codex_s56_20261030_1020')
OUT=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/probe_gt_dev_v1')
INIT=Path('/data_cf/cowmpo/init/libero_goal_dev3')
PROBES=Path('/data_cf/cowmpo/worlds/s6_strike/dev3')
EVIDENCE=Path('/data_cf/cowmpo/evidence/libero_goal_dev3_strike')

def worker(ep):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from cowm.init.gt_dump import NAME_MAP
    roles.require([ep],{'dev'},'Target probe GT diagnosis')
    man=json.loads((INIT/'manifest.json').read_text());info=man['episodes'][ep+'.npy']
    suite=benchmark.get_benchmark_dict()[man['suite']]();task=suite.get_task(info['task_id'])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                          camera_heights=256,camera_widths=256,ignore_done=True)
    env.reset();sim=env.env.sim;priv=np.load(INIT/info['privileged'])
    sim.model.body_pos[:]=priv['body_pos'];sim.model.body_quat[:]=priv['body_quat']
    obs=env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']])
    warm=np.zeros(7);warm[6]=-1
    for _ in range(man['num_steps_wait']):obs,_,_,_=env.step(warm)
    model=sim.model
    roots={}
    for name,prefix in NAME_MAP.items():
        roots[name]=next(b for b in range(model.nbody) if int(model.body_rootid[b])==b and
                         (model.body_id2name(b) or '').startswith(prefix+'_'))
    pr=json.loads((PROBES/ep/'probe.json').read_text());frames=set(pr['frames']);alignment=[]
    def save(step,obs):
        frame=f'{ep}__f{step:03d}';dest=OUT/frame;dest.mkdir()
        rgb=np.ascontiguousarray(obs['agentview_image'][::-1,::-1])
        expected=np.asarray(Image.open(EVIDENCE/frame/'frame0.png').convert('RGB'))
        alignment.append({'frame':frame,'rgb_equal':bool(np.array_equal(rgb,expected)),
                          'mean_abs_difference':float(np.abs(rgb.astype(int)-expected.astype(int)).mean())})
        seg=sim.render(width=256,height=256,camera_name='agentview',segmentation=True)
        gid=seg[...,1].astype(int);gid[(seg[...,0]!=5)|(gid<0)]=-1;gid=gid[::-1,::-1]
        root=np.where(gid>=0,model.body_rootid[model.geom_bodyid[np.clip(gid,0,None)]],-1)
        np.savez_compressed(dest/'gt_masks.npz',**{n:root==b for n,b in roots.items()})
    if 0 in frames:save(0,obs)
    for step,act in enumerate(pr['actions'],1):
        obs,_,_,_=env.step(np.asarray(act))
        if step in frames:save(step,obs)
    env.close();(OUT/(ep+'_alignment.json')).write_text(json.dumps(alignment,indent=1))
    print(ep,'aligned',sum(r['rgb_equal'] for r in alignment),len(alignment))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--episode');a=parser.parse_args()
    if a.episode:return worker(a.episode)
    OUT.mkdir(exist_ok=False)
    eps=sorted(p.name for p in PROBES.iterdir() if p.is_dir());roles.require(eps,{'dev'},'Target diagnosis')
    def run(ep):
        with (OUT/(ep+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--episode',ep],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((ep,result.returncode))
        print('target_replayed',ep,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(run,eps))
    records=[r for p in OUT.glob('*_alignment.json') for r in json.loads(p.read_text())]
    summary={'episodes':len(eps),'frames':len(records),'rgb_equal':sum(r['rgb_equal'] for r in records),
             'max_mean_abs_difference':max(r['mean_abs_difference'] for r in records)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':main()
