# Copyright 2026 The RLinf Authors.
"""Initial-RGB Panda estimation with per-channel appearance nuisance parameters."""
import argparse,hashlib,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion,gaussian_filter
from scipy.optimize import least_squares
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'robot_rgb_affine_dev_v1';WORLD=R/'bowl_support_dev_v1/world11';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worker(ep):
    dest=OUT/ep;dest.mkdir();scene=Scene(str(WORLD));scene.reset();source=R/'natural_states_all_dev_v1'/ep/'state.json';apply_state(scene,json.loads(source.read_text())['objects'],require_matched=True)
    robot=scene.env.robots[0];sim=scene.env.sim;idx=robot._ref_joint_pos_indexes;nominal=sim.data.qpos[idx].copy();roi=binary_dilation(scene.robot_mask(),iterations=5);interior=binary_erosion(scene.robot_mask(),iterations=2)
    masks=R/'visual_dev_bound_v1/dev3'/ep/'masks.npz'
    with np.load(masks) as a:occupied=np.logical_or.reduce([a[n] for n in a.files])
    roi&=~binary_dilation(occupied,iterations=2);interior&=roi
    image=Path('/data_cf/cowmpo/evidence/libero_goal_dev3')/ep/'frame0.png';target=gaussian_filter(np.asarray(Image.open(image).convert('RGB'),dtype=float)/255,(1.2,1.2,0));wanted=target[roi]
    baseline=R/'robot_rgb_all_dev_v2'/ep/'robot.json';record=json.loads(baseline.read_text())['robot'];initial_delta=np.asarray(record['arm_qpos'])-nominal;count=0;calibration=None
    def residual(delta):
        nonlocal count,calibration
        sim.data.qpos[idx]=nominal+delta;sim.data.qvel[robot._ref_joint_vel_indexes]=0;sim.forward();count+=1
        render=gaussian_filter(np.asarray(scene.rgb(),dtype=float)/255,(1.2,1.2,0));coeff=[]
        for c in range(3):
            x=render[...,c][interior];y=target[...,c][interior];matrix=np.stack([x,np.ones_like(x)],-1)
            gain,bias=np.linalg.lstsq(matrix,y,rcond=None)[0];gain=float(np.clip(gain,.5,2.));bias=float(np.clip(bias,-.3,.3));coeff.append([gain,bias])
        calibration=np.asarray(coeff);pred=render[roi]*calibration[:,0]+calibration[:,1]
        return (pred-wanted).reshape(-1)
    def jac(delta):
        cols=[]
        for j in range(7):
            d=np.zeros(7);d[j]=.003;cols.append((residual(delta+d)-residual(delta-d))/.006)
        return np.stack(cols,-1)
    initial=residual(initial_delta);fit=least_squares(residual,initial_delta,jac=jac,bounds=(-.08,.08),max_nfev=50,ftol=1e-5,xtol=1e-5,gtol=1e-5);final=residual(fit.x)
    record={**record,'arm_qpos':sim.data.qpos[idx].tolist()}
    report={'episode':ep,'robot':record,'rgb_rmse_before':float(np.sqrt(np.mean(initial**2))),'rgb_rmse_after':float(np.sqrt(np.mean(final**2))),'appearance_coefficients':calibration.tolist(),
        'render_evaluations':count,'success':bool(fit.success),'sources':{str(q):sha(q) for q in [source,baseline,masks,image]},'world_tree_sha256':tree_sha(WORLD),
        'protocol':'initial RGB only; eliminate affine gain/bias from fixed robot interior; qpos7 optimized, gripper/velocity/shared controller priors retained; no Target parameter input'}
    Image.fromarray(scene.rgb()).save(dest/'render.png');(dest/'robot.json').write_text(json.dumps(report,indent=1));scene.env.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--episode');a=p.parse_args()
    if a.episode:return worker(a.episode)
    OUT.mkdir(exist_ok=False);digest=tree_sha(WORLD);(OUT/'freeze.json').write_text(json.dumps({'episodes':EPS,'world_tree_sha256':digest,'script_sha256':sha(__file__),'GT_inputs':False,'scope':'dev-only estimator hypothesis, must pass geometry and action replay diagnosis'},indent=1))
    def run(ep):
        with (OUT/(ep+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--episode',ep],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((ep,r.returncode))
    with ThreadPoolExecutor(6) as pool:list(pool.map(run,EPS))
    verify_tree(WORLD,digest);print('ROBOT_RGB_AFFINE_COMPLETE',flush=True)
if __name__=='__main__':main()
