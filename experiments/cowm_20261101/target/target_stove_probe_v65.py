# Copyright 2026 The RLinf Authors.
"""v65: execute the 12 stove twist-direction fit probes (stove_probe_v65/freeze.json); public RGB only, prefix images verified."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
from builder_batch import OUT,sha,write

B=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/stove_probe_v65')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/stove_probe_v65')
E=Path('/data_cf/cowmpo/evidence/codex_stove_probe_v65')

def worker(index):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    fz=json.loads((B/'freeze.json').read_text());key=fz['queries'][index]
    source=B/(key+'_query.json');q=json.loads(source.read_text())
    if sha(source)!=fz['query_sha256'][key]:raise ValueError('query changed')
    trial=int(q['episode'].split('_i')[-1])
    suite=benchmark.get_benchmark_dict()['libero_goal']();task=suite.get_task(7);envs=[];observations=[]
    for _ in range(3):
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
        env.seed(0);envs.append(env)
    for i,env in enumerate(envs):
        env.env.sim._render_context_offscreen.gl_ctx.make_current();env.reset();env.set_init_state(suite.get_task_init_states(7)[i])
        for _ in range(15):obs,_,_,_=env.step(np.zeros(7))
        observations.append(obs)
    env=envs[trial];obs=observations[trial]
    with np.load(q['public_image_source']) as a:expected=a['frames'].copy();old_steps=a['frame_idx'].copy();old_actions=a['actions'].copy()
    if not np.array_equal(old_actions[:q['prefix_steps']],np.asarray(q['actions'][:q['prefix_steps']])):raise ValueError('action prefix mismatch')
    frames=[];checks=0
    for step in range(len(q['actions'])+1):
        env.env.sim._render_context_offscreen.gl_ctx.make_current()
        if step:obs,_,_,_=env.step(np.asarray(q['actions'][step-1]))
        rgb=np.ascontiguousarray(obs['agentview_image'][::-1,::-1])
        if step<=q['prefix_steps'] and step in old_steps:
            if not np.array_equal(rgb,expected[int(np.where(old_steps==step)[0][0])]):raise ValueError('original prefix RGB mismatch')
            checks+=1
        if step in q['frames']:frames.append(rgb)
    if len(frames)!=len(q['frames']):raise ValueError('incomplete public frames')
    dest=E/q['key'];dest.mkdir()
    np.savez_compressed(dest/'public_stream.npz',frames=frames,frame_idx=q['frames'],actions=q['actions'])
    Image.fromarray(frames[-1]).save(dest/'final.png')
    write(dest/'query.json',{'episode':q['episode'],'query_sha256':sha(source),'prefix_exact_frames':checks,'frames':q['frames'],'GT_exported':False})
    write(T/(q['key']+'.json'),{'prefix_exact_frames':checks,'public_frames':len(frames),'query_sha256':sha(source)})
    for env in envs:env.env.sim._render_context_offscreen.gl_ctx.make_current();env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int);a=p.parse_args()
    if a.index is not None:return worker(a.index)
    doc=json.loads((B/'freeze.json').read_text())
    if len(doc['queries'])!=12:raise ValueError('expected twelve frozen queries')
    T.mkdir(exist_ok=False);E.mkdir(exist_ok=False)
    write(T/'protocol.json',{'freeze_sha256':sha(B/'freeze.json'),'queries':12,'GT_exported':False})
    for i in range(12):
        with (T/(str(i)+'.log')).open('w') as log:subprocess.run([sys.executable,__file__,'--index',str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('STOVE_V65_TARGET_COMPLETE',i+1,'/12',flush=True)
    write(T/'complete.json',{'queries':12,'all_prefix_checks_exact':True})

if __name__=='__main__':main()
