# Copyright 2026 The RLinf Authors.
"""Fit a generic supported box jointly across six dev masks using known camera geometry."""
import hashlib
import itertools
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.special import expit
from cowm import roles
from cowm.init.estimate_state import Scene,iou
from cowm.init.aabb import object_aabb
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'cheese_rgb_support_dev_v2';WORLD=R/'stove_parts_geometry_dev_v1/world07'
EPS=[f'libero_goal_t{k}_i0' for k in range(6)];roles.require(EPS,{'dev'},'supported box geometry development')
OUT.mkdir(exist_ok=False);(OUT/'freeze.json').write_text(json.dumps({'episodes':EPS,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'world_tree_sha256':tree_sha(WORLD),'source':'bound RGB masks only; common table support hypothesis'},indent=1))
scene=Scene(str(WORLD));scene.reset();meta=scene.world;ob=next(o for o in meta['objects'] if o['name']=='cream_cheese');dims0=np.array([ob['attrs'][n] for n in ['size_x','size_y','size_z']]);top=meta['table']['height']
m,d=scene.env.sim.model,scene.env.sim.data;cid=m.camera_name2id('agentview');rot=d.cam_xmat[cid].reshape(3,3).copy();camera=d.cam_xpos[cid].copy();f=128/np.tan(np.radians(m.cam_fovy[cid])/2)
signs=np.array(list(itertools.product([-.5,.5],[-.5,.5],[0.,1.])))
records=[]
for ep in EPS:
    doc=json.loads((R/'stove_parts_states_dev_v1'/ep/'state.json').read_text());old=json.loads(json.dumps(doc['objects']['cream_cheese']))
    with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:mask=a['cream_cheese'].copy()
    ys,xs=np.nonzero(mask);yy,xx=np.mgrid[max(0,ys.min()-6):min(256,ys.max()+7),max(0,xs.min()-6):min(256,xs.max()+7)]
    records.append({'ep':ep,'doc':doc,'old':old,'mask':mask,'points':np.stack([xx.ravel(),yy.ravel()],-1),'target':gaussian_filter(mask.astype(float),.5)[yy,xx].ravel()})
def polygon(dims,xy,yaw):
    c,s=np.cos(yaw),np.sin(yaw);r=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    p=(signs*dims)@r.T+np.array([*xy,top]);p=(p-camera)@rot;depth=-p[:,2]
    points=np.stack([255-(128+f*p[:,0]/depth),128-f*p[:,1]/depth],axis=-1)
    return cv2.convexHull(points.astype(np.float32)).reshape(-1,2).astype(float)
def occupancy(poly,points):
    edges=np.roll(poly,-1,axis=0)-poly;deltas=points[:,None,:]-poly[None,:,:]
    distance=(edges[None,:,0]*deltas[:,:,1]-edges[None,:,1]*deltas[:,:,0])/np.linalg.norm(edges,axis=-1)[None,:]
    return expit(distance.min(axis=1)/.4)
def residual(x):
    dims=dims0*np.exp(x[:3]);out=[]
    for i,rec in enumerate(records):
        dx,dy,dyaw=x[3+3*i:6+3*i];xy=np.array(rec['old']['pos'][:2])+np.array([dx,dy])*.01;yaw=rec['old']['yaw']+.2*dyaw
        out.extend(occupancy(polygon(dims,xy,yaw),rec['points'])-rec['target'])
    return np.r_[out,.01*x[:3]]
x0=np.zeros(3+3*len(EPS));lo=np.r_[np.log([.6,.6,.5]),np.tile([-3,-3,-2.5],len(EPS))];hi=np.r_[np.log([1.4,1.4,1.8]),np.tile([3,3,2.5],len(EPS))]
def jac(x):
    step=.002;cols=[]
    for j in range(len(x)):
        dx=np.zeros_like(x);dx[j]=step;cols.append((residual(x+dx)-residual(x-dx))/(2*step))
    return np.stack(cols,axis=1)
fit=least_squares(residual,x0,bounds=(lo,hi),jac=jac,max_nfev=180,ftol=1e-7,xtol=1e-7,gtol=1e-7)
dims=dims0*np.exp(fit.x[:3]);world=OUT/'world';shutil.copytree(WORLD,world);meta=json.loads((world/'world.json').read_text());ob=next(o for o in meta['objects'] if o['name']=='cream_cheese');tree=ET.parse(world/ob['xml']);scale=dims/dims0
for g in tree.iter('geom'):
    for key in ['size','pos']:g.set(key,' '.join(map(str,np.array(list(map(float,g.get(key).split())))*scale)))
tree.write(world/ob['xml']);ob['attrs'].update(zip(['size_x','size_y','size_z'],dims.tolist()));ob['top_z']=float(dims[2]);ob['pos'][2]=float(top)
meta.pop('world_sha256');meta['experimental_geometry']={'model':'RGB multi-episode supported generic box','episodes':EPS,'Target_input':False};meta['world_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();(world/'world.json').write_text(json.dumps(meta,indent=1));scene.env.close()
scene=Scene(str(world));scores=[]
for i,rec in enumerate(records):
    scene.reset();state=rec['doc']['objects'];dx,dy,dyaw=fit.x[3+3*i:6+3*i];xy=np.array(rec['old']['pos'][:2])+np.array([dx,dy])*.01;yaw=rec['old']['yaw']+.2*dyaw
    state['cream_cheese'].update(pos=[*xy,float(top)],yaw=float(yaw));apply_state(scene,state,require_matched=True)
    k=scene.names.index('cream_cheese');seg=scene.seg();c,h=object_aabb(scene.env.sim.model,scene.env.sim.data,np.nonzero(scene.geom2obj==k)[0]);state['cream_cheese'].update(aabb_center_rel_base=(c-scene.base).tolist(),aabb_half=h.tolist())
    dest=OUT/'states'/rec['ep'];dest.mkdir(parents=True);rec['doc'].update(objects=state,world_sha256=meta['world_sha256'],world_tree_sha256=tree_sha(world));(dest/'state.json').write_text(json.dumps(rec['doc'],indent=1))
    with np.load(R/'visual_dev_bound_v1/dev3'/rec['ep']/'masks.npz') as a:np.savez_compressed(dest/'masks.npz',**{n:a[n] for n in a.files})
    np.savez_compressed(dest/'render_masks.npz',**{n:seg==j for j,n in enumerate(scene.names)});scores.append(iou(seg==k,rec['mask']))
scene.env.close();report={'dims':dims.tolist(),'sam_iou':scores,'mean_sam_iou':float(np.mean(scores)),'initial_loss':float(np.mean(residual(x0)**2)),'fitted_loss':float(np.mean(fit.fun**2)),'success':bool(fit.success),'source':'RGB bound masks + shared camera/table prior, no Target geometry'};(OUT/'report.json').write_text(json.dumps(report,indent=1));print(json.dumps(report),flush=True)
