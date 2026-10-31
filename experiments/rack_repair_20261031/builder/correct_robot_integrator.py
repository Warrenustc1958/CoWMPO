# Copyright 2026 The RLinf Authors.
"""Derive corrected RGB state records from the shared Panda's own post-warmup integrator."""
import hashlib
import json
import shutil
from pathlib import Path
from cowm.init.estimate_state import Scene
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SOURCE=R/'robot_rgb_all_dev_v1';OUT=R/'robot_rgb_all_dev_v2';OUT.mkdir(exist_ok=False)
scene=Scene(str(R/'bowl_support_dev_v1/world11'));scene.reset();integrator=scene.env.robots[0].gripper.current_action.tolist()
if len(integrator)!=2:raise ValueError('unexpected Panda actuator-integrator dimension')
for p in sorted(SOURCE.glob('*/robot.json')):
    ep=p.parent.name;dest=OUT/ep;shutil.copytree(p.parent,dest);doc=json.loads(p.read_text());doc['robot']['grip_current_action']=integrator
    doc['integrator_correction']={'parent_record_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source':'shared CodeWorld Panda reset plus 15 warmup steps; no Target state','reason':'Panda policy action dimension 1; internal actuator integrator dimension 2'}
    (dest/'robot.json').write_text(json.dumps(doc,indent=1))
scene.env.close();(OUT/'freeze.json').write_text(json.dumps({'source':str(SOURCE),'episodes':32,'integrator':integrator,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'Target_input':False},indent=1));print('RGB_ROBOT_INTEGRATOR_CORRECTED',32,integrator,flush=True)
