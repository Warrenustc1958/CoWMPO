# Copyright 2026 The RLinf Authors.
"""Target-private attribution of which objects actually constrain the rack probes."""
import argparse,hashlib,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm.init.gt_dump import NAME_MAP
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');OUT=T/'rack_contact_attribution_v1';EP='libero_goal_t0_i0';PLAN=R/'rack_multidir_preflight_dev_v1'
def worker(key):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    init=Path('/data_cf/cowmpo/init/libero_goal_dev3');man=json.loads((init/'manifest.json').read_text());info=man['episodes'][EP+'.npy'];suite=benchmark.get_benchmark_dict()[man['suite']]();task=suite.get_task(info['task_id']);env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),use_camera_obs=False,has_offscreen_renderer=False,ignore_done=True);env.reset();sim=env.env.sim;m=sim.model
    with np.load(init/info['privileged']) as a:m.body_pos[:]=a['body_pos'];m.body_quat[:]=a['body_quat']
    env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']]);warm=np.zeros(7);warm[6]=-1
    for _ in range(man['num_steps_wait']):env.step(warm)
    rroot=m.body_name2id(env.env.robots[0].robot_model.root_body);robots=np.array([int(m.body_rootid[m.geom_bodyid[g]])==rroot or (m.body_id2name(int(m.geom_bodyid[g])) or '').startswith(('robot0','gripper0')) for g in range(m.ngeom)]);names={}
    for n,prefix in NAME_MAP.items():
        for b in range(m.nbody):
            if int(m.body_rootid[b])==b and (m.body_id2name(b) or '').startswith(prefix+'_'):names[b]=n
    probe=json.loads((PLAN/key/'probe.json').read_text());rows=[]
    for step,act in enumerate(probe['actions'],1):
        env.step(np.asarray(act));hits=set();gripper=set()
        for c in sim.data.contact[:sim.data.ncon]:
            for rg,og in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                if not robots[rg] or robots[og]:continue
                root=int(m.body_rootid[m.geom_bodyid[og]]);name=names.get(root,'table' if 'table' in (m.geom_id2name(og) or '') else 'other');hits.add(name)
                if (m.body_id2name(int(m.geom_bodyid[rg])) or '').startswith('gripper0'):gripper.add(name)
        rows.append({'step':step,'phase':next(p for p,last in probe['phases'].items() if step<=last),'contacts':sorted(hits),'gripper_contacts':sorted(gripper),'max_arm_speed':float(np.max(np.abs(sim.data.qvel[env.env.robots[0]._ref_joint_vel_indexes])))})
    env.close();report={'key':key,'query_sha256':hashlib.sha256((PLAN/key/'probe.json').read_bytes()).hexdigest(),'rows':rows};(OUT/(key+'.json')).write_text(json.dumps(report,indent=1))
    summary={'key':key,'phases':{}}
    for phase in probe['phases']:
        rr=[r for r in rows if r['phase']==phase];summary['phases'][phase]={'steps':len(rr),'object_contact_steps':{n:sum(n in r['contacts'] for r in rr) for n in sorted({n for r in rr for n in r['contacts']})},'gripper_contact_steps':{n:sum(n in r['gripper_contacts'] for r in rr) for n in sorted({n for r in rr for n in r['gripper_contacts']})},'max_arm_speed':max(r['max_arm_speed'] for r in rr)}
    (OUT/(key+'_summary.json')).write_text(json.dumps(summary,indent=1))
def main():
    p=argparse.ArgumentParser();p.add_argument('--key');a=p.parse_args()
    if a.key:return worker(a.key)
    OUT.mkdir(exist_ok=False);keys=json.loads((PLAN/'selection.json').read_text())['selected']
    def run(key):
        with (OUT/(key+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',key],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(('Target contact diagnostic failed',key))
        return json.loads((OUT/(key+'_summary.json')).read_text())
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,keys))
    (OUT/'summary.json').write_text(json.dumps(rows,indent=1));print(json.dumps(rows,indent=1),flush=True)
if __name__=='__main__':main()
