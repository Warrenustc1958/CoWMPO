# Copyright 2026 The RLinf Authors.
"""Bind fit grasp RGB to verified cheese and shared-robot anchors, with no private labels."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'grasp_tracking_ready_fit_v2';OUT.mkdir(exist_ok=False);EPROOT=R/'natural_bowl_fit_v2';EPS=sorted(p.parent.name for p in EPROOT.glob('*/state.json'));roles.require(EPS,{'fit'},'grasp response binding');scene=Scene(str(R/'bowl_support_dev_v1/world11'));sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for mode in ['grasp','miss']:
    root=OUT/mode
    for ep in EPS:
        scene.reset();doc=json.loads((EPROOT/ep/'state.json').read_text());robot_file=R/'robot_rgb_fit_v2'/ep/'robot.json';robot=json.loads(robot_file.read_text())['robot'];apply_state(scene,doc['objects'],require_matched=True,robot=robot)
        anchor=root/'anchors'/ep;anchor.mkdir(parents=True)
        source=R/'functional_complete_fit_v2'/ep/'masks.npz'
        with np.load(source) as a:cheese=a['cream_cheese']
        np.savez_compressed(anchor/'masks.npz',cream_cheese=cheese,robot=scene.robot_mask());(anchor/'association.json').write_text(json.dumps({'objects':{n:{'status':'matched'} for n in ['cream_cheese','robot']},'masks_sha256':sha(anchor/'masks.npz'),'initial_identity_source_sha256':sha(source),'robot_record_sha256':sha(robot_file)}))
        key=ep+'__'+mode;probe_path=R/'grasp_contrast_fit_v2'/key/'probe.json';probe=json.loads(probe_path.read_text());dest=root/'probes'/ep;dest.mkdir(parents=True);shutil.copy2(probe_path,dest/'probe.json')
        public=Path('/data_cf/cowmpo/evidence/codex_grasp_contrast_fit_v2')/key
        for step in probe['frames']:
            src=public/f'f{step:03d}'/'frame0.png';target=root/'evidence'/f'{ep}__f{step:03d}';target.mkdir(parents=True);shutil.copy2(src,target/'frame0.png');sources[str(src)]=sha(src)
        first=root/'initial'/ep;first.mkdir(parents=True);shutil.copy2(public/'f000/frame0.png',first/'frame0.png')
scene.env.close();(OUT/'freeze.json').write_text(json.dumps({'episodes':EPS,'role':'fit','queries':56,'images':len(sources),'sources':sources,'script_sha256':sha(Path(__file__)),'Target_private_inputs':False},indent=1));print('GRASP_FIT_VIDEO_READY',len(EPS),len(sources),flush=True)
