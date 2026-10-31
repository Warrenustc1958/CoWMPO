# Copyright 2026 The RLinf Authors.
"""Target-private quality audit of grasp video evidence, with exact RGB replay checks."""
import argparse,hashlib,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from cowm import roles
from cowm.init.gt_dump import NAME_MAP

def main():
    p=argparse.ArgumentParser()
    for n in ['tracking','probes','evidence','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--role',choices=['dev','fit'],required=True);p.add_argument('--episodes',default='');p.add_argument('--episode');p.add_argument('--expected-episodes',type=int,required=True);a=p.parse_args()
    eps=a.episodes.split(',') if a.episodes else sorted(q.parent.name.removesuffix('__grasp') for q in a.probes.glob('*__grasp/probe.json'));roles.require(eps,{a.role},'Target video diagnostic')
    if a.episode:return worker(a,a.episode)
    if len(eps)!=a.expected_episodes:raise ValueError('wrong exact episode set')
    a.out.mkdir(exist_ok=False)
    def run(ep):
        args=[sys.executable,__file__,'--role',a.role,'--episode',ep,'--expected-episodes',str(a.expected_episodes)]
        if a.episodes:args+=['--episodes',a.episodes]
        for n in ['tracking','probes','evidence','out']:args+=['--'+n,str(getattr(a,n))]
        with (a.out/(ep+'.log')).open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Target diagnostic worker failed; inspect private log')
        return json.loads((a.out/(ep+'.json')).read_text())
    with ThreadPoolExecutor(8) as pool:rows=list(pool.map(run,eps))
    totals={n:{k:sum(r['objects'][n][k] for r in rows) for k in rows[0]['objects'][n] if k!='iou_values'} for n in ['robot','cream_cheese']}
    for n in totals:
        values=[v for r in rows for v in r['objects'][n]['iou_values']];totals[n].update(mean_matched_iou=float(np.mean(values)) if values else None,min_matched_iou=min(values) if values else None)
    report={'episodes':len(eps),'role':a.role,'RGB_replay_exact':all(r['RGB_exact'] for r in rows),'objects':totals,'scope':'Target-private GT diagnostic; fit exports aggregates only; identity is best matching object at IoU>=.3, not a segmentation guarantee'};(a.out/'summary.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)

def worker(a,ep):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    roles.require([ep],{a.role},'Target video diagnostic');key=ep+'__grasp';probe=json.loads((a.probes/key/'probe.json').read_text());init=Path('/data_cf/cowmpo/init')/('libero_goal_dev3' if roles.parse(ep)[1] in [0,1,2] else 'libero_goal_fit3');man=json.loads((init/'manifest.json').read_text());info=man['episodes'][ep+'.npy'];suite=benchmark.get_benchmark_dict()[man['suite']]();task=suite.get_task(info['task_id']);env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True);env.reset();sim=env.env.sim;m=sim.model
    with np.load(init/info['privileged']) as priv:m.body_pos[:]=priv['body_pos'];m.body_quat[:]=priv['body_quat']
    obs=env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']]);warm=np.zeros(7);warm[6]=-1
    for _ in range(man['num_steps_wait']):obs,_,_,_=env.step(warm)
    rootmap={}
    for n,pre in NAME_MAP.items():
        ids=[b for b in range(m.nbody) if int(m.body_rootid[b])==b and (m.body_id2name(b) or '').startswith(pre+'_')]
        if ids:rootmap[n]=ids[0]
    robot_geom=np.array([(m.body_id2name(int(m.geom_bodyid[g])) or '').startswith(('robot0','gripper0')) for g in range(m.ngeom)])
    counts={n:{'expected':0,'matched':0,'identity_correct':0,'iou_ge_05':0,'missing_visible':0,'missing_invisible':0,'false_visible':0,'iou_values':[]} for n in ['robot','cream_cheese']};sources={}
    def score(step,obs):
        rgb=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);image=a.evidence/key/f'f{step:03d}'/'frame0.png'
        if not np.array_equal(rgb,np.asarray(Image.open(image).convert('RGB'))):raise ValueError('Target RGB replay differs from frozen evidence')
        raw=sim.render(width=256,height=256,camera_name='agentview',segmentation=True);gid=raw[...,1].astype(np.int64);gid[(raw[...,0]!=5)|(gid<0)]=-1;gid=gid[::-1,::-1];groot=np.where(gid>=0,m.body_rootid[m.geom_bodyid[np.clip(gid,0,None)]],-1);gt={n:groot==b for n,b in rootmap.items()};gt['robot']=(gid>=0)&robot_geom[np.clip(gid,0,None)];frame=f'{ep}__f{step:03d}';path=a.tracking/frame/'masks.npz';association=json.loads((path.parent/'association.json').read_text());sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as arr:
            for n,c in counts.items():
                status=association['objects'][n]['status'];c['expected']+=1;visible=bool(gt[n].any())
                if status!='matched':c['missing_visible' if visible else 'missing_invisible']+=1;continue
                pred=arr[n];c['matched']+=1
                if not visible:c['false_visible']+=1
                scores={k:float((pred&mk).sum()/(pred|mk).sum()) if (pred|mk).any() else 0. for k,mk in gt.items()};iou=scores[n];c['identity_correct']+=int(iou>=.3 and max(scores,key=scores.get)==n);c['iou_ge_05']+=int(iou>=.5);c['iou_values'].append(iou)
    frames=set(probe['frames'])
    if 0 in frames:score(0,obs)
    for step,act in enumerate(probe['actions'],1):
        obs,_,_,_=env.step(np.asarray(act))
        if step in frames:score(step,obs)
    env.close();(a.out/(ep+'.json')).write_text(json.dumps({'episode':ep,'RGB_exact':True,'objects':counts,'sources':sources},indent=1))
if __name__=='__main__':main()
