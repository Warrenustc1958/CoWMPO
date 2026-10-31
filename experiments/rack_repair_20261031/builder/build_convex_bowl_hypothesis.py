# Copyright 2026 The RLinf Authors.
"""Generic collision topology contrast, using only the code world's own mesh and inertia."""
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from cowm.init.estimate_state import Scene
from cowm.world.artifact import tree_sha
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');BASE=R/'cheese_rgb_support_dev_v2/world';OUT=R/'bowl_convex_contact_dev_v1'
OUT.mkdir(exist_ok=False);world=OUT/'world';shutil.copytree(BASE,world)
scene=Scene(str(BASE));scene.reset();k=scene.names.index('bowl');m=scene.env.sim.model;bid=m.body_name2id(scene.env.objects[k].root_body)
inertial={'pos':' '.join(map(str,m.body_ipos[bid])),'quat':' '.join(map(str,m.body_iquat[bid])),'mass':str(float(m.body_mass[bid])),'diaginertia':' '.join(map(str,m.body_inertia[bid]))}
if float(inertial['mass'])<=0:raise ValueError('missing root body inertia')
meta=json.loads((world/'world.json').read_text());ob=meta['objects'][k];tree=ET.parse(world/ob['xml']);root=tree.getroot();body=root.find('.//body[@name="object"]')
visual=next(g for g in body.findall('geom') if g.get('group')=='1' and g.get('type')=='mesh');collisions=[g for g in body.findall('geom') if g.get('group','0')=='0']
if not collisions:raise ValueError('no bowl collision parts')
new=ET.Element('geom',dict(collisions[0].attrib));new.set('mesh',visual.get('mesh'))
for g in collisions:body.remove(g)
body.append(new)
if body.find('inertial') is not None:raise ValueError('explicit inertia already exists')
body.insert(0,ET.Element('inertial',inertial));tree.write(world/ob['xml'])
meta.pop('world_sha256');meta['experimental_contact_topology']={'object':'bowl','hypothesis':'single convex envelope of own visual mesh','inertia':'frozen to baseline compiled mass and inertia','Target_input':False}
meta['world_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();(world/'world.json').write_text(json.dumps(meta,indent=1));scene.env.close()
check=Scene(str(world));check.reset();m2=check.env.sim.model;b2=m2.body_name2id(check.env.objects[k].root_body)
report={'world_tree_sha256':tree_sha(world),'parent_world_tree_sha256':tree_sha(BASE),'old_collision_parts':len(collisions),'new_collision_parts':1,'mass_difference':float(m2.body_mass[b2])-float(inertial['mass']),'inertia_difference':(m2.body_inertia[b2]-list(map(float,inertial['diaginertia'].split()))).tolist(),'source':'generic collision envelope; only own frozen mesh and compiled inertia'}
check.env.close();(OUT/'freeze.json').write_text(json.dumps(report,indent=1));print(json.dumps(report),flush=True)
