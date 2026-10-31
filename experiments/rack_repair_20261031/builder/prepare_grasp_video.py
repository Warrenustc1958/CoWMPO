# Copyright 2026 The RLinf Authors.
"""Prepare grasp RGB and shared-robot/verified-object anchors, without Target internals."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'grasp_tracking_ready_dev_v1'
OUT.mkdir(exist_ok=False)
scene=Scene(str(R/'stove_parts_geometry_dev_v1/world07'))
for mode in ['grasp','miss']:
    for i in range(6):
        ep=f'libero_goal_t{i}_i0';key=ep+'__'+mode;root=OUT/mode
        scene.reset();state=json.loads((R/'stove_parts_states_dev_v1'/ep/'state.json').read_text())['objects'];apply_state(scene,state,require_matched=True)
        anchor=root/'anchors'/ep;anchor.mkdir(parents=True)
        with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:cheese=a['cream_cheese']
        np.savez_compressed(anchor/'masks.npz',cream_cheese=cheese,robot=scene.robot_mask())
        (anchor/'association.json').write_text(json.dumps({'objects':{n:{'status':'matched'} for n in ['cream_cheese','robot']},'masks_sha256':hashlib.sha256((anchor/'masks.npz').read_bytes()).hexdigest()}))
        probe=json.loads((R/'grasp_contrast_dev_v1'/key/'probe.json').read_text())
        dest=root/'probes'/ep;dest.mkdir(parents=True);(dest/'probe.json').write_text(json.dumps(probe))
        public=Path('/data_cf/cowmpo/evidence/codex_grasp_contrast_dev_v1')/key
        for step in probe['frames']:
            target=root/'evidence'/f'{ep}__f{step:03d}';target.mkdir(parents=True)
            shutil.copy2(public/f'f{step:03d}'/'frame0.png',target/'frame0.png')
        initial=root/'initial'/ep;initial.mkdir(parents=True);shutil.copy2(public/'f000/frame0.png',initial/'frame0.png')
scene.env.close();print('GRASP_VIDEO_READY',flush=True)
