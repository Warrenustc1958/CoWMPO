# Copyright 2026 The RLinf Authors.
"""Fit shared Panda kinematics to image feature correspondences, using initial RGB only."""
import argparse,hashlib,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,binary_dilation
from scipy.optimize import least_squares
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'robot_rgb_feature_dev_v2';WORLD=R/'bowl_support_dev_v1/world11';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
def get_real_depth_map(sim,depth):
    if not (np.all(depth>=0) and np.all(depth<=1)):raise ValueError('normalized MuJoCo depth required')
    extent=sim.model.stat.extent;near=sim.model.vis.map.znear*extent;far=sim.model.vis.map.zfar*extent
    return near/(1.-depth*(1.-near/far))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worker(ep):
    dest=OUT/ep;dest.mkdir();scene=Scene(str(WORLD));scene.reset();source=R/'natural_states_all_dev_v1'/ep/'state.json';baseline=R/'robot_rgb_all_dev_v2'/ep/'robot.json';record=json.loads(baseline.read_text())['robot'];apply_state(scene,json.loads(source.read_text())['objects'],require_matched=True,robot=record)
    robot=scene.env.robots[0];sim=scene.env.sim;m,d=sim.model,sim.data;idx=robot._ref_joint_pos_indexes;q0=np.asarray(record['arm_qpos']);cid=m.camera_name2id('agentview');camera=d.cam_xpos[cid].copy();rot=d.cam_xmat[cid].reshape(3,3).copy();f=128/np.tan(np.radians(m.cam_fovy[cid])/2)
    image=Path('/data_cf/cowmpo/evidence/libero_goal_dev3')/ep/'frame0.png';target=np.asarray(Image.open(image).convert('RGB'));target_gray=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
    moving=np.zeros(m.ngeom,bool);arm_dofs=set(robot._ref_joint_vel_indexes)
    for g in range(m.ngeom):
        b=int(m.geom_bodyid[g])
        while b>0:
            if set(range(int(m.body_dofadr[b]),int(m.body_dofadr[b]+m.body_dofnum[b])))&arm_dofs:moving[g]=True;break
            b=int(m.body_parentid[b])
    masks=R/'visual_dev_bound_v1/dev3'/ep/'masks.npz'
    with np.load(masks) as a:occupied=np.logical_or.reduce([a[n] for n in a.files])
    rows=[];delta=np.zeros(7)
    for iteration in range(3):
        sim.data.qpos[idx]=q0+delta;sim.forward();rgb=scene.rgb();gid=scene._seg_raw();roi=(gid>=0)&moving[np.clip(gid,0,None)];roi=binary_erosion(roi,iterations=2)&~binary_dilation(occupied,iterations=2)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);points=cv2.goodFeaturesToTrack(gray,maxCorners=100,qualityLevel=.01,minDistance=4,mask=(roi*255).astype(np.uint8),blockSize=3)
        if points is None or len(points)<8:raise ValueError('insufficient moving-robot features')
        p1,status,error=cv2.calcOpticalFlowPyrLK(gray,target_gray,points,None,winSize=(11,11),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
        back,bstatus,_=cv2.calcOpticalFlowPyrLK(target_gray,gray,p1,None,winSize=(11,11),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
        p0=points[:,0];p1=p1[:,0];fb=np.linalg.norm(back[:,0]-p0,axis=1);valid=(status[:,0]>0)&(bstatus[:,0]>0)&(fb<=.5)&(np.linalg.norm(p1-p0,axis=1)<=8)
        scene.env._make_render_context_current();_,depthraw=sim.render(width=256,height=256,camera_name='agentview',depth=True);depth=get_real_depth_map(sim,depthraw)[::-1,::-1]
        local=[];bids=[];observed=[];source_points=[];correlations=[]
        for j in np.nonzero(valid)[0]:
            x,y=p0[j];tx,ty=p1[j]
            if min(x,y,tx,ty)<5 or max(x,y,tx,ty)>250:continue
            patch0=cv2.getRectSubPix(gray,(7,7),(float(x),float(y))).astype(float);patch1=cv2.getRectSubPix(target_gray,(7,7),(float(tx),float(ty))).astype(float)
            z0=patch0-patch0.mean();z1=patch1-patch1.mean();corr=float((z0*z1).sum()/max(np.linalg.norm(z0)*np.linalg.norm(z1),1e-9))
            if corr<.8:continue
            px,py=int(round(x)),int(round(y));g=int(gid[py,px])
            if g<0 or not moving[g]:continue
            z=float(depth[py,px]);pc=np.array([(255-x-128)*z/f,-(y-128)*z/f,-z]);world=camera+rot@pc;bid=int(m.geom_bodyid[g]);lp=(world-d.body_xpos[bid])@d.body_xmat[bid].reshape(3,3)
            local.append(lp);bids.append(bid);observed.append([tx,ty]);source_points.append([x,y]);correlations.append(corr)
        if len(local)<8:raise ValueError('insufficient verified feature correspondences')
        local=np.asarray(local);bids=np.asarray(bids);observed=np.asarray(observed)
        def project(x):
            sim.data.qpos[idx]=q0+x;sim.forward();world=np.einsum('ni,nji->nj',local,d.body_xmat[bids].reshape(-1,3,3))+d.body_xpos[bids];pc=(world-camera)@rot;z=-pc[:,2]
            return np.stack([255-(128+f*pc[:,0]/z),128-f*pc[:,1]/z],-1)
        if np.max(np.abs(project(delta)-source_points))>1e-4:raise ValueError('feature unprojection/reprojection mismatch')
        def residual(x):return np.r_[(project(x)-observed).reshape(-1),.1*x/.08]
        before=residual(delta);fit=least_squares(residual,delta,bounds=(-.08,.08),loss='soft_l1',f_scale=.5,diff_step=1e-3,max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8);delta=fit.x;after=residual(delta)
        rows.append({'iteration':iteration,'features':len(local),'mean_correlation':float(np.mean(correlations)),'reprojection_rmse_before':float(np.sqrt(np.mean(before[:-7]**2))),'reprojection_rmse_after':float(np.sqrt(np.mean(after[:-7]**2))),'success':bool(fit.success)})
    sim.data.qpos[idx]=q0+delta;sim.forward();record={**record,'arm_qpos':sim.data.qpos[idx].tolist()};Image.fromarray(scene.rgb()).save(dest/'render.png');report={'episode':ep,'robot':record,'iterations':rows,'sources':{str(q):sha(q) for q in [source,baseline,masks,image]},'world_tree_sha256':tree_sha(WORLD),'protocol':'initial RGB forward/backward LK features on shared moving robot; known depth and kinematics; no Target state or GT masks'}
    (dest/'robot.json').write_text(json.dumps(report,indent=1));scene.env.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--episode');a=p.parse_args()
    if a.episode:return worker(a.episode)
    OUT.mkdir(exist_ok=False);digest=tree_sha(WORLD);(OUT/'freeze.json').write_text(json.dumps({'episodes':EPS,'world_tree_sha256':digest,'script_sha256':sha(__file__),'GT_inputs':False,'scope':'six-dev geometric estimator hypothesis; requires physical validation'},indent=1))
    def run(ep):
        with (OUT/(ep+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--episode',ep],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((ep,r.returncode))
    with ThreadPoolExecutor(6) as pool:list(pool.map(run,EPS))
    verify_tree(WORLD,digest);print('ROBOT_FEATURE_DEV_COMPLETE',flush=True)
if __name__=='__main__':main()
