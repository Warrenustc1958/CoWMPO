# Copyright 2026 The RLinf Authors.
"""Test gripper-weighted initial-RGB fitting using the shared Panda model only."""
import argparse,hashlib,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,gaussian_filter
from scipy.optimize import least_squares
from cowm.init.estimate_state import Scene
from cowm.init.robot_state import robot_issues
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'robot_rgb_gripper_dev_v1';WORLD=R/'bowl_support_dev_v1/world11';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worker(ep):
    dest=OUT/ep;dest.mkdir();scene=Scene(str(WORLD));scene.reset();source=R/'natural_states_all_dev_v1'/ep/'state.json';old=R/'robot_rgb_all_dev_v2'/ep/'robot.json'
    record=json.loads(old.read_text())['robot'];apply_state(scene,json.loads(source.read_text())['objects'],require_matched=True,robot=record)
    robot=scene.env.robots[0];sim=scene.env.sim;m=sim.model;arm=np.asarray(robot._ref_joint_pos_indexes);grip=np.asarray(robot._ref_gripper_joint_pos_indexes);ids=np.r_[arm,grip];q0=sim.data.qpos[ids].copy()
    image=Path('/data_cf/cowmpo/evidence/libero_goal_dev3')/ep/'frame0.png';target=np.asarray(Image.open(image).convert('RGB'),dtype=float)/255
    gid=scene._seg_raw();gripper_geom=np.array([(m.body_id2name(int(m.geom_bodyid[g])) or '').startswith('gripper0') for g in range(m.ngeom)])
    gripper=(gid>=0)&gripper_geom[np.clip(gid,0,None)];roi=binary_dilation(scene.robot_mask(),iterations=5)
    mask=R/'visual_dev_bound_v1/dev3'/ep/'masks.npz'
    with np.load(mask) as data:occupied=np.logical_or.reduce([data[n] for n in data.files])
    roi&=~binary_dilation(occupied,iterations=2);local=binary_dilation(gripper,iterations=6)&roi
    if local.sum()<30:raise ValueError('insufficient visible gripper region')
    weight=np.where(local[roi],np.sqrt(6.),1.);wanted=gaussian_filter(target,(.7,.7,0))[roi]
    lower=np.r_[np.full(7,-.025),np.full(2,-.012)];upper=-lower
    for j,idx in enumerate(ids):
        joint=int(np.nonzero(m.jnt_qposadr==idx)[0][0])
        if m.jnt_limited[joint]:
            lower[j]=max(lower[j],float(m.jnt_range[joint,0])-q0[j]);upper[j]=min(upper[j],float(m.jnt_range[joint,1])-q0[j])
    count=0
    def residual(delta):
        nonlocal count
        sim.data.qpos[ids]=q0+delta;sim.forward();count+=1
        rgb=np.asarray(scene.rgb(),dtype=float)/255
        return ((gaussian_filter(rgb,(.7,.7,0))[roi]-wanted)*weight[:,None]).reshape(-1)
    def jac(delta):
        cols=[]
        for j in range(9):
            step=.0015 if j<7 else .0005;v=np.zeros(9);v[j]=step;cols.append((residual(delta+v)-residual(delta-v))/(2*step))
        return np.stack(cols,-1)
    initial=residual(np.zeros(9));fit=least_squares(residual,np.zeros(9),jac=jac,bounds=(lower,upper),max_nfev=40,ftol=1e-5,xtol=1e-5,gtol=1e-5);final=residual(fit.x)
    record={**record,'arm_qpos':sim.data.qpos[arm].tolist(),'grip_qpos':sim.data.qpos[grip].tolist()}
    if robot_issues(record):raise ValueError(robot_issues(record))
    report={'episode':ep,'robot':record,'weighted_rgb_rmse_before':float(np.sqrt(np.mean(initial**2))),'weighted_rgb_rmse_after':float(np.sqrt(np.mean(final**2))),
        'local_pixels':int(local.sum()),'render_evaluations':count,'success':bool(fit.success),'delta':fit.x.tolist(),'sources':{str(p):sha(p) for p in [source,old,image,mask]},
        'world_tree_sha256':tree_sha(WORLD),'protocol':'initial RGB only; gripper region weight6; sigma .7; arm bound .025rad, finger bound .012m within model limits; no Target input'}
    Image.fromarray(scene.rgb()).save(dest/'render.png');(dest/'robot.json').write_text(json.dumps(report,indent=1));scene.env.close();print(ep,report['weighted_rgb_rmse_before'],report['weighted_rgb_rmse_after'],flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--episode');a=p.parse_args()
    if a.episode:return worker(a.episode)
    OUT.mkdir(exist_ok=False);digest=tree_sha(WORLD);(OUT/'freeze.json').write_text(json.dumps({'episodes':EPS,'world_tree_sha256':digest,'script_sha256':sha(__file__),'GT_inputs':False,'scope':'six-dev estimator hypothesis, not an accepted replacement'},indent=1))
    def run(ep):
        with (OUT/(ep+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--episode',ep],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((ep,r.returncode))
    with ThreadPoolExecutor(6) as pool:list(pool.map(run,EPS))
    verify_tree(WORLD,digest);print('RGB_GRIPPER_DEV_COMPLETE',flush=True)
if __name__=='__main__':main()
