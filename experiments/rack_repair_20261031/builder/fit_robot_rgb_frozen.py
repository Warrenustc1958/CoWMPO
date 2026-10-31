# Copyright 2026 The RLinf Authors.
"""Apply the frozen initial-RGB Panda estimator without Target state inputs."""
import argparse,hashlib,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,gaussian_filter
from scipy.optimize import least_squares
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
from cowm.init.robot_state import robot_issues

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(a):
    ep=a.episode;roles.require([ep],{a.role},'frozen RGB Panda estimation');dest=a.out/ep;dest.mkdir()
    scene=Scene(str(a.world));scene.reset();state=a.states/ep/'state.json'
    apply_state(scene,json.loads(state.read_text())['objects'],require_matched=True)
    robot=scene.env.robots[0];sim=scene.env.sim;idx=robot._ref_joint_pos_indexes;nominal=sim.data.qpos[idx].copy()
    image=a.evidence/ep/'frame0.png';mask=a.masks/ep/'masks.npz'
    target=np.asarray(Image.open(image).convert('RGB'),dtype=float)/255
    roi=binary_dilation(scene.robot_mask(),iterations=5)
    with np.load(mask) as arr:occupied=np.logical_or.reduce([arr[n] for n in arr.files])
    roi &= ~binary_dilation(occupied,iterations=2)
    if roi.sum()<100:raise ValueError('insufficient robot pixels for frozen estimator')
    wanted=gaussian_filter(target,(1.2,1.2,0))[roi];count=0
    def residual(delta):
        nonlocal count
        sim.data.qpos[idx]=nominal+delta;sim.data.qvel[robot._ref_joint_vel_indexes]=0;sim.forward();count+=1
        rgb=np.asarray(scene.rgb(),dtype=float)/255
        return (gaussian_filter(rgb,(1.2,1.2,0))[roi]-wanted).reshape(-1)
    def jac(delta):
        columns=[]
        for j in range(7):
            d=np.zeros(7);d[j]=.003;columns.append((residual(delta+d)-residual(delta-d))/.006)
        return np.stack(columns,axis=-1)
    initial=residual(np.zeros(7));fit=least_squares(residual,np.zeros(7),jac=jac,bounds=(-.08,.08),max_nfev=60,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    final=residual(fit.x)
    record={'arm_qpos':(nominal+fit.x).tolist(),'arm_qvel':[0.]*7,'grip_qpos':sim.data.qpos[robot._ref_gripper_joint_pos_indexes].tolist(),'grip_qvel':[0.]*2,'grip_current_action':robot.gripper.current_action.tolist()}
    if robot_issues(record):raise ValueError(robot_issues(record))
    report={'episode':ep,'robot':record,'initial_rgb_rmse':float(np.sqrt(np.mean(initial**2))),'fitted_rgb_rmse':float(np.sqrt(np.mean(final**2))),'success':bool(fit.success),'render_evaluations':count,'roi_pixels':int(roi.sum()),'sources':{str(p):sha(p) for p in [state,image,mask]},'world_tree_sha256':tree_sha(a.world),'protocol':'frozen first-six-dev photometric estimator; zero velocity prior; post-reset shared actuator integrator; no Target state'}
    Image.fromarray(scene.rgb()).save(dest/'render.png');(dest/'robot.json').write_text(json.dumps(report,indent=1));scene.env.close()

def main():
    p=argparse.ArgumentParser()
    for n in ['world','states','masks','evidence','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--role',choices=['dev','fit'],required=True);p.add_argument('--episode');p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    if a.episode:return worker(a)
    eps=sorted(q.parent.name for q in a.states.glob('*/state.json'));roles.require(eps,{a.role},'frozen RGB Panda estimation');a.out.mkdir(exist_ok=False)
    digest=tree_sha(a.world);(a.out/'freeze.json').write_text(json.dumps({'episodes':eps,'role':a.role,'world_hash':digest,'script_sha256':sha(Path(__file__)),'estimator_config':'gaussian1.2 ROI dilation5 mask exclusion2 finite difference.003 bounds.08 max_nfev60','GT_inputs':False},indent=1))
    def run(ep):
        args=[sys.executable,__file__,'--episode',ep,'--role',a.role]
        for n in ['world','states','masks','evidence','out']:args+=['--'+n,str(getattr(a,n))]
        with (a.out/(ep+'.log')).open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((ep,r.returncode))
        print('RGB_ROBOT_FROZEN_COMPLETE',ep,flush=True)
    with ThreadPoolExecutor(a.jobs) as pool:list(pool.map(run,eps))
    verify_tree(a.world,digest);print('RGB_ROBOT_SET_COMPLETE',len(eps),flush=True)
if __name__=='__main__':main()
