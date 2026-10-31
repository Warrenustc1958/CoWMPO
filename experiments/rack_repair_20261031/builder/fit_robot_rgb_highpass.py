# Copyright 2026 The RLinf Authors.
"""Estimate small deviations from the shared Panda reset using only the initial RGB."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,gaussian_filter
from scipy.optimize import least_squares
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'robot_rgb_highpass_dev_v1'
EPS=[f'libero_goal_t{k}_i0' for k in range(6)];WORLD=R/'stove_parts_geometry_dev_v1/world07'
roles.require(EPS,{'dev'},'robot RGB estimator development')
def worker(ep):
    dest=OUT/ep;dest.mkdir();scene=Scene(str(WORLD));scene.reset()
    apply_state(scene,json.loads((R/'stove_parts_states_dev_v1'/ep/'state.json').read_text())['objects'],require_matched=True)
    robot=scene.env.robots[0];sim=scene.env.sim;idx=robot._ref_joint_pos_indexes;nominal=sim.data.qpos[idx].copy()
    source=Path('/data_cf/cowmpo/evidence/libero_goal_dev3')/ep/'frame0.png'
    target=np.asarray(Image.open(source).convert('RGB'),dtype=float)/255
    roi=binary_dilation(scene.robot_mask(),iterations=5)
    with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:
        occupied=np.logical_or.reduce([a[n] for n in a.files])
    roi &= ~binary_dilation(occupied,iterations=2)
    wanted=(gaussian_filter(target,(1.2,1.2,0))-gaussian_filter(target,(4.,4.,0)))[roi]
    count=0
    def residual(delta):
        nonlocal count
        sim.data.qpos[idx]=nominal+delta;sim.data.qvel[robot._ref_joint_vel_indexes]=0;sim.forward()
        rgb=np.asarray(scene.rgb(),dtype=float)/255;count+=1
        return ((gaussian_filter(rgb,(1.2,1.2,0))-gaussian_filter(rgb,(4.,4.,0)))[roi]-wanted).reshape(-1)
    def jac(delta):
        step=.003;cols=[]
        for j in range(7):
            d=np.zeros(7);d[j]=step;cols.append((residual(delta+d)-residual(delta-d))/(2*step))
        return np.stack(cols,axis=-1)
    initial=residual(np.zeros(7));initial_rgb=scene.rgb();initial_mask=scene.robot_mask()
    fit=least_squares(residual,np.zeros(7),jac=jac,bounds=(-.08,.08),max_nfev=60,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    final=residual(fit.x);Image.fromarray(scene.rgb()).save(dest/'render.png');Image.fromarray(initial_rgb).save(dest/'initial_render.png')
    np.savez_compressed(dest/'masks.npz',initial=initial_mask,fitted=scene.robot_mask())
    report={'episode':ep,'robot':{'arm_qpos':(nominal+fit.x).tolist(),'arm_qvel':[0.]*7,'grip_qpos':sim.data.qpos[robot._ref_gripper_joint_pos_indexes].tolist(),'grip_qvel':[0.]*len(robot._ref_gripper_joint_vel_indexes),'grip_current_action':[-1.]},
      'initial_rgb_rmse':float(np.sqrt(np.mean(initial**2))),'fitted_rgb_rmse':float(np.sqrt(np.mean(final**2))),'delta_norm_rad':float(np.linalg.norm(fit.x)),
      'render_evaluations':count,'success':bool(fit.success),'roi_pixels':int(roi.sum()),'world_tree_sha256':tree_sha(WORLD),'image_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'protocol':'initial RGB only; objective highpass; fixed ROI, gaussian sigma1.2, joint bounds +/-0.08 rad, qvel assumed zero; no Target state input'}
    (dest/'robot.json').write_text(json.dumps(report,indent=1));scene.env.close()
    print(ep,report['initial_rgb_rmse'],report['fitted_rgb_rmse'],count,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--episode');a=p.parse_args()
    if a.episode:return worker(a.episode)
    OUT.mkdir(exist_ok=False);(OUT/'freeze.json').write_text(json.dumps({'episodes':EPS,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source':'initial RGB only; no Target state'},indent=1))
    def run(ep):
        with (OUT/(ep+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--episode',ep],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((ep,r.returncode))
    with ThreadPoolExecutor(6) as pool:list(pool.map(run,EPS))
    print('ROBOT_RGB_FIT_COMPLETE',flush=True)
if __name__=='__main__':main()
