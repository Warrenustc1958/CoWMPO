# Copyright 2026 The RLinf Authors.
"""Target-private observability diagnostic with perfect robot silhouettes; never a Builder estimator."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation,gaussian_filter
from scipy.optimize import least_squares
from cowm.init.estimate_state import Scene,iou
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');OUT=T/'robot_mask_ceiling_v1';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
def worker(ep):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    init=Path('/data_cf/cowmpo/init/libero_goal_dev3');man=json.loads((init/'manifest.json').read_text());info=man['episodes'][ep+'.npy'];suite=benchmark.get_benchmark_dict()[man['suite']]();task=suite.get_task(info['task_id'])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True);env.reset();sim=env.env.sim;m=sim.model
    with np.load(init/info['privileged']) as a:m.body_pos[:]=a['body_pos'];m.body_quat[:]=a['body_quat']
    env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']]);warm=np.zeros(7);warm[6]=-1
    for _ in range(man['num_steps_wait']):env.step(warm)
    raw=sim.render(width=256,height=256,camera_name='agentview',segmentation=True);gid=raw[...,1].astype(int);gid[(raw[...,0]!=5)|(gid<0)]=-1;gid=gid[::-1,::-1]
    geoms=np.array([(m.body_id2name(int(m.geom_bodyid[g])) or '').startswith(('robot0','gripper0')) for g in range(m.ngeom)]);target=(gid>=0)&geoms[np.clip(gid,0,None)];env.close()
    with np.load(T/'robot_initial_dev_v1'/(ep+'.npz')) as a:gt={n:a[n] for n in a.files}
    scene=Scene(str(R/'bowl_support_dev_v1/world11'));scene.reset();state=json.loads((R/'natural_states_all_dev_v1'/ep/'state.json').read_text())['objects'];apply_state(scene,state,require_matched=True)
    robot=scene.env.robots[0];sim=scene.env.sim;idx=robot._ref_joint_pos_indexes;nominal=sim.data.qpos[idx].copy();roi=binary_dilation(scene.robot_mask()|target,iterations=5);wanted=gaussian_filter(target.astype(float),1.2)[roi]
    record=json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot'];initial_delta=np.asarray(record['arm_qpos'])-nominal;count=0
    def residual(delta):
        nonlocal count
        sim.data.qpos[idx]=nominal+delta;sim.forward();count+=1
        return gaussian_filter(scene.robot_mask().astype(float),1.2)[roi]-wanted
    def jac(delta):
        cols=[]
        for j in range(7):
            d=np.zeros(7);d[j]=.003;cols.append((residual(delta+d)-residual(delta-d))/.006)
        return np.stack(cols,-1)
    initial=residual(initial_delta);before=float(np.linalg.norm(scene.env.eef_pos()-gt['eef'])*1000)
    oracle=residual(gt['arm_q']-nominal);oracle_iou=iou(scene.robot_mask(),target)
    fit=least_squares(residual,initial_delta,jac=jac,bounds=(-.08,.08),max_nfev=60,ftol=1e-5,xtol=1e-5,gtol=1e-5);final=residual(fit.x);after=float(np.linalg.norm(scene.env.eef_pos()-gt['eef'])*1000)
    report={'episode':ep,'eef_before_mm':before,'eef_after_mm':after,'initial_mask_rmse':float(np.sqrt(np.mean(initial**2))),'fitted_mask_rmse':float(np.sqrt(np.mean(final**2))),
        'oracle_mask_rmse':float(np.sqrt(np.mean(oracle**2))),'oracle_render_iou':oracle_iou,'fitted_render_iou':iou(scene.robot_mask(),target),'success':bool(fit.success),'render_evaluations':count,
        'private_fitted_arm_qpos':sim.data.qpos[idx].tolist()}
    scene.env.close();(OUT/(ep+'.json')).write_text(json.dumps(report,indent=1));np.savez_compressed(OUT/(ep+'_gt.npz'),robot=target)
def main():
    p=argparse.ArgumentParser();p.add_argument('--episode');a=p.parse_args()
    if a.episode:return worker(a.episode)
    OUT.mkdir(exist_ok=False)
    def run(ep):
        with (OUT/(ep+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--episode',ep],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(('private mask diagnostic failed',ep))
        return json.loads((OUT/(ep+'.json')).read_text())
    with ThreadPoolExecutor(6) as pool:rows=list(pool.map(run,EPS))
    report={'episodes':6,'mean_eef_before_mm':float(np.mean([r['eef_before_mm'] for r in rows])),'mean_eef_after_mm':float(np.mean([r['eef_after_mm'] for r in rows])),
        'max_eef_after_mm':max(r['eef_after_mm'] for r in rows),'within_2mm':sum(r['eef_after_mm']<2 for r in rows),'mean_oracle_render_iou':float(np.mean([r['oracle_render_iou'] for r in rows])),
        'mean_fitted_render_iou':float(np.mean([r['fitted_render_iou'] for r in rows])),'scope':'perfect-mask observability diagnostic only; all inferred joint records and GT masks remain Target-private'}
    (OUT/'summary.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)
if __name__=='__main__':main()
