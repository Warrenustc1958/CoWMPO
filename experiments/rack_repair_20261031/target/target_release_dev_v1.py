# Copyright 2026 The RLinf Authors.
"""Execute frozen release actions; labels and contact trajectories remain Target-private."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from cowm import roles
from plan_release_dev_v1 import classify_release
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/release_controls_dev_v1')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/release_controls_dev_v1')
E=Path('/data_cf/cowmpo/evidence/codex_release_controls_dev_v1')
INIT=Path('/data_cf/cowmpo/init/libero_goal_dev3')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(key):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    source=R/key/'probe.json';probe=json.loads(source.read_text());ep=probe['episode'];roles.require([ep],{'dev'},'release controls Target')
    manifest=json.loads((INIT/'manifest.json').read_text());info=manifest['episodes'][ep+'.npy']
    suite=benchmark.get_benchmark_dict()[manifest['suite']]();task=suite.get_task(info['task_id'])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
    env.reset();sim=env.env.sim;m=sim.model
    with np.load(INIT/info['privileged']) as a:m.body_pos[:]=a['body_pos'];m.body_quat[:]=a['body_quat']
    obs=env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']]);warm=np.zeros(7);warm[6]=-1
    for _ in range(manifest['num_steps_wait']):obs,_,_,_=env.step(warm)
    root=next(b for b in range(m.nbody) if int(m.body_rootid[b])==b and (m.body_id2name(b) or '').startswith('cream_cheese_'))
    rroot=m.body_name2id(env.env.robots[0].robot_model.root_body)
    robot=np.array([int(m.body_rootid[m.geom_bodyid[g]])==rroot or (m.body_id2name(int(m.geom_bodyid[g])) or '').startswith(('robot0','gripper0')) for g in range(m.ngeom)])
    def contacts():
        fingers=set();rob=False;support=False
        for c in sim.data.contact[:sim.data.ncon]:
            for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                if int(m.body_rootid[m.geom_bodyid[g]])!=root:continue
                if robot[r]:
                    rob=True
                    if 'finger' in (m.geom_id2name(r) or ''):fingers.add(int(m.geom_bodyid[r]))
                elif int(m.body_rootid[m.geom_bodyid[r]])!=root:support=True
        return len(fingers),rob,support
    initial=sim.data.body_xpos[root].copy();trajectory=[];rc=[];sp=[];dest=E/key;dest.mkdir();frames=set(probe['frames'])
    def save(step):
        d=dest/f'f{step:03d}';d.mkdir();Image.fromarray(np.ascontiguousarray(obs['agentview_image'][::-1,::-1])).save(d/'frame0.png')
    save(0)
    for step,act in enumerate(probe['actions'],1):
        obs,_,_,_=env.step(np.asarray(act))
        if step in frames:save(step)
        if step==probe['phases']['holding']:held=sim.data.body_xpos[root].copy();fingers=contacts()[0]
        if step==probe['phases']['before_release']:release=sim.data.body_xpos[root].copy()
        if step>probe['phases']['before_release']:
            trajectory.append(sim.data.body_xpos[root].copy());_,r,s=contacts();rc.append(r);sp.append(s)
    label=classify_release(initial,held,fingers,release,trajectory,rc,sp);env.close()
    result={'key':key,'episode':ep,'label':label,'query_sha256':sha(source),'frames':len(frames),'predictions':{}}
    for variant in ['shape','reference']:
        pred=json.loads((R/key/(variant+'.json')).read_text())
        if pred['query_sha256']!=sha(source):raise ValueError('prediction/action mismatch')
        result['predictions'][variant]=pred['prediction']
    (T/(key+'.json')).write_text(json.dumps(result,indent=1))
    np.savez_compressed(T/(key+'_trajectory.npz'),positions=np.asarray(trajectory),robot_contact=rc,support=sp)
    (dest/'query.json').write_text(json.dumps({'key':key,'episode':ep,'query_sha256':sha(source),'frames':probe['frames']},indent=1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--key');a=p.parse_args()
    if a.key:return worker(a.key)
    frozen=json.loads((R/'freeze.json').read_text());complete=json.loads((R/'complete.json').read_text());keys=frozen['keys']
    if len(keys)!=12 or keys!=complete['keys']:raise ValueError('wrong exact release query set')
    for k,h in complete['query_hashes'].items():
        if sha(R/k/'probe.json')!=h:raise ValueError('query changed')
    T.mkdir(exist_ok=False);E.mkdir(exist_ok=False)
    (T/'protocol.json').write_text(json.dumps({'plan_freeze_sha256':sha(R/'freeze.json'),'complete_sha256':sha(R/'complete.json'),'script_sha256':sha(__file__),'labels_private':True,'role':'dev'},indent=1))
    def run(key):
        with (T/(key+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',key],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((key,r.returncode))
        print('TARGET_RELEASE_DONE',key,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(run,keys))
    rows=[json.loads((T/(k+'.json')).read_text()) for k in keys]
    report={'queries':len(rows),'frames':sum(r['frames'] for r in rows),'labels':dict(Counter(r['label']['status'] for r in rows)),'by_mode':{},'models':{}}
    for mode in ['low','high']:report['by_mode'][mode]=dict(Counter(r['label']['status'] for r in rows if r['key'].endswith('__'+mode)))
    for model in ['shape','reference']:
        confusion=Counter((r['label']['status'],r['predictions'][model]['status']) for r in rows)
        recalls={label:sum(r['label']['status']==label and r['predictions'][model]['status']==label for r in rows)/sum(r['label']['status']==label for r in rows) for label in ['stable','dropped'] if any(r['label']['status']==label for r in rows)}
        report['models'][model]={'confusion':[{'label':a,'prediction':b,'n':n} for (a,b),n in sorted(confusion.items())],'conditional_release_recalls':recalls,
            'conditional_release_ba':float(np.mean(list(recalls.values()))) if len(recalls)==2 else None,'all_query_accuracy':sum(r['label']['status']==r['predictions'][model]['status'] for r in rows)/len(rows)}
    report['scope']='six exposed dev initial states; low/high release controls; no_carry and unresolved retained; conditional BA is not independent full P1 acceptance'
    (T/'summary.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)
if __name__=='__main__':main()
