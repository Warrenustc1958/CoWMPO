# Copyright 2026 The RLinf Authors.
"""Track the shared robot prior through public rack-touch RGB; no GT."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');EP='libero_goal_t0_i0'
OUT=R/'rack_touch_tracking_ready_v1';OUT.mkdir(exist_ok=False)
for key in ['h070','h170','h270']:
    root=OUT/key;anchor=root/'anchors'/EP;anchor.mkdir(parents=True)
    predicted=R/'rack_touch_dev_v1'/key/'predicted_frames'/f'{EP}__f000'
    with np.load(predicted/'masks.npz') as a:robot=a['robot']
    np.savez_compressed(anchor/'masks.npz',robot=robot)
    record={'schema':1,'episode':EP,'objects':{'robot':{'status':'matched','identity_source':'shared Panda reset model'}},
        'masks_sha256':hashlib.sha256((anchor/'masks.npz').read_bytes()).hexdigest()}
    (anchor/'association.json').write_text(json.dumps(record,indent=1))
    initial=root/'initial'/EP;initial.mkdir(parents=True)
    shutil.copy2(Path('/data_cf/cowmpo/evidence/codex_rack_touch_dev_v1')/key/f'{EP}__f000'/'frame0.png',initial/'frame0.png')
print('RACK_TOUCH_VIDEO_READY')
