# Copyright 2026 The RLinf Authors.
"""Generic package contact-shape contrast selected from frozen RGB action responses."""
import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import trimesh
from cowm.init.estimate_state import Scene,iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha
from replay_rgb_robot import install_robot
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');BASE=R/'bowl_support_dev_v1/world11';OUT=R/'package_contact_dev_v2'
RADII=[0.,.002,.004,.006];EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
def build(index,inertia):
    radius=RADII[index];world=OUT/f'world{index:02d}';shutil.copytree(BASE,world)
    meta=json.loads((world/'world.json').read_text());ob=next(o for o in meta['objects'] if o['name']=='cream_cheese');dims=np.array([ob['attrs'][n] for n in ['size_x','size_y','size_z']]);tree=ET.parse(world/ob['xml']);body=tree.find('.//body[@name="object"]');geom=next(g for g in body.findall('geom') if g.get('group')=='0')
    if radius:
        points=[]
        for sign in itertools.product([-1,1],repeat=3):
            sign=np.array(sign)
            for axis in range(3):
                p=sign*(dims/2-radius);p[axis]+=sign[axis]*radius;p[2]+=dims[2]/2;points.append(p)
        mesh=trimesh.convex.convex_hull(np.asarray(points));mesh.export(world/'assets/cream_cheese/chamfer_contact.obj')
        asset=tree.getroot().find('asset');ET.SubElement(asset,'mesh',name='package_chamfer_contact',file='chamfer_contact.obj')
        for key in ['size','pos']:geom.attrib.pop(key,None)
        geom.set('type','mesh');geom.set('mesh','package_chamfer_contact')
    body.insert(0,ET.Element('inertial',inertia));tree.write(world/ob['xml']);meta.pop('world_sha256')
    meta['experimental_package_contact']={'shape':'generic chamfered cuboid','radius':radius,'inertia':'frozen code-world baseline','selection':'six dev RGB responses only','Target_input':False}
    meta['world_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();(world/'world.json').write_text(json.dumps(meta,indent=1));return world
def worker(index):
    world=OUT/f'world{index:02d}';dest=OUT/f'replay{index:02d}';dest.mkdir();scene=Scene(str(world));scores=[];rows=[]
    for ep in EPS:
        scene.reset();apply_state(scene,json.loads((R/'natural_states_all_dev_v1'/ep/'state.json').read_text())['objects'],require_matched=True);install_robot(scene,json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot'])
        p=json.loads((R/'grasp_contrast_dev_v1'/(ep+'__grasp')/'probe.json').read_text());k=scene.names.index('cream_cheese');initial=scene.env.object_pos('cream_cheese').copy();fingers=set();series=[]
        for step,act in enumerate(p['actions'],1):
            scene.env.step(np.asarray(act))
            if 190<=step<=240 and step%10==0:
                with np.load(R/'grasp_video_dev_v1/grasp'/f'{ep}__f{step:03d}'/'masks.npz') as a:target=a['cream_cheese']
                mask=scene.seg()==k;score=iou(mask,target);scores.append(score);series.append(score)
            if step==p['phases']['holding']:
                held=scene.env.object_pos('cream_cheese').copy();sim=scene.env.sim
                for c in sim.data.contact[:sim.data.ncon]:
                    for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                        if scene.geom2obj[g]==k and 'finger' in (sim.model.geom_id2name(r) or ''):fingers.add(int(sim.model.geom_bodyid[r]))
        rows.append({'episode':ep,'prediction':bool(held[2]-initial[2]>.05 and len(fingers)>=2),'response_iou':series})
    scene.env.close();report={'index':index,'radius':RADII[index],'mean_response_iou':float(np.mean(scores)),'world':str(world),'world_tree_sha256':tree_sha(world),'rows':rows};(dest/'result.json').write_text(json.dumps(report,indent=1));print(json.dumps(report),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int);a=p.parse_args()
    if a.index is not None:return worker(a.index)
    OUT.mkdir(exist_ok=False);scene=Scene(str(BASE));scene.reset();k=scene.names.index('cream_cheese');m=scene.env.sim.model;bid=m.body_name2id(scene.env.objects[k].root_body)
    inertia={'pos':' '.join(map(str,m.body_ipos[bid])),'quat':' '.join(map(str,m.body_iquat[bid])),'mass':str(float(m.body_mass[bid])),'diaginertia':' '.join(map(str,m.body_inertia[bid]))};scene.env.close()
    for i in range(len(RADII)):build(i,inertia)
    evidence={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'grasp_video_dev_v1/grasp').glob('*/masks.npz')}
    (OUT/'freeze.json').write_text(json.dumps({'radii':RADII,'episodes':EPS,'evidence':evidence,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'objective':'mean cream_cheese SAM2 response IoU steps190..240; all episodes retained','Target_input':False},indent=1))
    def run(i):
        with (OUT/f'replay{i:02d}.log').open('w') as log:r=subprocess.run([sys.executable,__file__,'--index',str(i)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((i,r.returncode))
        return json.loads((OUT/f'replay{i:02d}/result.json').read_text())
    with ThreadPoolExecutor(4) as pool:reports=list(pool.map(run,range(len(RADII))))
    best=max(reports,key=lambda r:r['mean_response_iou']);(OUT/'selection.json').write_text(json.dumps({'best':best,'candidates':reports},indent=1));print('PACKAGE_CONTACT_SELECTED',best['index'],best['mean_response_iou'],flush=True)
if __name__=='__main__':main()
