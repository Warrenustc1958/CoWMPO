# Copyright 2026 The RLinf Authors.
"""Fit a generic two-shelf rack silhouette with shared camera and support plane only."""
import argparse,hashlib,itertools,json,math,shutil,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np
from scipy.special import expit
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from cowm import roles
from cowm.init.estimate_state import Scene
from experimental_rack import build
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'rack_supported_silhouette_dev_v1';BASE=R/'bowl_support_dev_v1/world11';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
SIGNS=np.array(list(itertools.product([-1,1],repeat=3)))

def parts(width,depth,height,incline):
    thickness=.008;slope=np.sin(incline);co=np.cos(incline);upper=height-.45*depth*abs(slope)-thickness;lower=max(.05,.45*upper)
    rows=[(np.array([depth/2,width/2,.004]),np.array([0,0,.004]),0.)]
    for level in [lower,upper]:
        for j in range(8):
            x=(-.45+.9*j/7)*depth;rows.append((np.array([.035*depth,width/2,thickness/2]),np.array([co*x,0,level-slope*x]),incline))
        rows.append((np.array([.012,width/2,.006]),np.array([0,0,level-.012]),0.))
    for side in [-1,1]:rows.append((np.array([.008,.008,height/2]),np.array([.25*depth,side*.35*width,height/2]),0.))
    return rows

def worker(seed):
    roles.require(EPS,{'dev'},'rack supported silhouette development');dest=OUT/f'candidate{seed:02d}';dest.mkdir()
    scene=Scene(str(BASE));scene.reset();m,d=scene.env.sim.model,scene.env.sim.data;cid=m.camera_name2id('agentview');rot=d.cam_xmat[cid].reshape(3,3).copy();camera=d.cam_xpos[cid].copy();f=128/np.tan(np.radians(m.cam_fovy[cid])/2);top=scene.world['table']['height'];scene.env.close()
    records=[]
    for ep in EPS:
        source=R/'stove_parts_states_dev_v1'/ep/'state.json';doc=json.loads(source.read_text())
        if ep=='libero_goal_t0_i0':initial=json.loads((R/'rack_touch_fit_dev_v3/candidate24/state.json').read_text())['objects']['rack']
        else:initial=json.loads((R/'rack_geometry_dev_v1/state04'/ep/'state.json').read_text())['objects']['rack']
        with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:mask=a['rack'].copy();other=np.logical_or.reduce([a[n] for n in a.files if n!='rack'])
        yy,xx=np.mgrid[0:256:3,0:256:3];points=np.stack([xx.ravel(),yy.ravel()],-1);weight=(~other[yy,xx]).ravel().astype(float)
        target=gaussian_filter(mask.astype(float),.5)[yy,xx].ravel();records.append({'ep':ep,'doc':doc,'old':initial,'points':points,'target':target,'weight':weight})
    dims0=np.array([.16992,.1687827456,.3094803936]);x0=np.r_[np.log(dims0),[.05,.4,.8][seed%3],np.zeros(3*len(EPS))]
    if seed>=3:x0[:3]+=np.log([1.5,1.,1.])
    lo=np.r_[np.log([.08,.07,.12]),0.,np.tile([-.20,-.20,-1.2],len(EPS))];hi=np.r_[np.log([.45,.40,.5]),1.1,np.tile([.20,.20,1.2],len(EPS))]
    def project(half,pos,pitch,xy,yaw):
        c,s=np.cos(pitch),np.sin(pitch);rp=np.array([[c,0,s],[0,1,0],[-s,0,c]]);c,s=np.cos(yaw),np.sin(yaw);rz=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        pts=((SIGNS*half)@rp.T+pos)@rz.T+np.array([*xy,top]);p=(pts-camera)@rot;dep=-p[:,2];uv=np.stack([255-(128+f*p[:,0]/dep),128-f*p[:,1]/dep],-1)
        return cv2.convexHull(uv.astype(np.float32)).reshape(-1,2).astype(float)
    def residual(x):
        width,depth,height=np.exp(x[:3]);boxes=parts(width,depth,height,x[3]);out=[]
        for i,rec in enumerate(records):
            dx,dy,dyaw=x[4+3*i:7+3*i];xy=np.asarray(rec['old']['pos'][:2])+[dx,dy];yaw=rec['old']['yaw']+dyaw;occupancy=np.zeros(len(rec['points']))
            for half,pos,pitch in boxes:
                poly=project(half,pos,pitch,xy,yaw);edge=np.roll(poly,-1,axis=0)-poly;delta=rec['points'][:,None,:]-poly[None,:,:]
                distance=(edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0])/np.maximum(np.linalg.norm(edge,axis=-1)[None,:],1e-8)
                occupancy=np.maximum(occupancy,expit(distance.min(axis=1)/.5))
            out.extend((occupancy-rec['target'])*rec['weight'])
        return np.r_[out,.03*(x[:3]-np.log(dims0))]
    def jac(x):
        cols=[]
        for j in range(len(x)):
            dx=np.zeros_like(x);dx[j]=.001 if j<4 else .0005;cols.append((residual(x+dx)-residual(x-dx))/(2*dx[j]))
        return np.stack(cols,axis=1)
    fit=least_squares(residual,x0,jac=jac,bounds=(lo,hi),max_nfev=65,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    width,depth,height=np.exp(fit.x[:3]);world=dest/'world';shutil.copytree(BASE,world);meta=json.loads((world/'world.json').read_text());ob=next(o for o in meta['objects'] if o['name']=='rack');ob['attrs'].update(width=float(width),depth=float(depth),height=float(height),incline=float(fit.x[3]));ob['top_z']=float(height);ob['pos'][2]=top
    build('rack',world/'assets/rack',width=width,depth=depth,height=height,incline=fit.x[3]);meta.pop('world_sha256');meta['rack_rgb_support_fit']={'source':'generic two-shelf model; six dev masks; shared camera/table; no Target geometry','seed':seed};meta['world_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();(world/'world.json').write_text(json.dumps(meta,indent=1))
    for i,rec in enumerate(records):
        dx,dy,dyaw=fit.x[4+3*i:7+3*i];s=rec['doc']['objects']['rack'];s.update(pos=[rec['old']['pos'][0]+dx,rec['old']['pos'][1]+dy,top],yaw=float(rec['old']['yaw']+dyaw));p=dest/'states'/rec['ep'];p.mkdir(parents=True);(p/'state.json').write_text(json.dumps(rec['doc'],indent=1))
    report={'seed':seed,'dims':[float(width),float(depth),float(height)],'incline':float(fit.x[3]),'loss':float(np.mean(fit.fun**2)),'success':bool(fit.success),'nfev':fit.nfev,'world':str(world)};(dest/'fit.json').write_text(json.dumps(report,indent=1));print('RACK_STATIC_SUPPORT_FIT',json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);a=p.parse_args()
    if a.seed is not None:return worker(a.seed)
    OUT.mkdir(exist_ok=False);(OUT/'freeze.json').write_text(json.dumps({'episodes':EPS,'seeds':6,'bounds':{'width':[.08,.45],'depth':[.07,.40],'height':[.12,.5],'incline':[0.,1.1]},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'Target_inputs':False,'role':'dev geometry candidate generation only; must validate contact and 3D separately'},indent=1))
    def run(seed):
        with (OUT/f'seed{seed}.log').open('w') as log:r=subprocess.run([sys.executable,__file__,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((seed,r.returncode))
        print('RACK_STATIC_CANDIDATE_DONE',seed,flush=True)
    with ThreadPoolExecutor(6) as pool:list(pool.map(run,range(6)))
if __name__=='__main__':main()
