# Copyright 2026 The RLinf Authors.
"""Target-only quality diagnostic for the shared-robot video evidence."""
import json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/rack_touch_video_v1')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/rack_touch_dev_v1')
report={}
for key in ['h070','h170','h270']:
    values=[]
    for path in sorted((R/key).glob('*/association.json')):
        with np.load(path.parent/'masks.npz') as a:pred=a['robot']
        with np.load(T/key/path.parent.name/'gt_masks.npz') as a:gt=a['robot']
        values.append(float((pred&gt).sum()/(pred|gt).sum()))
    if len(values)!=28:raise ValueError('incomplete robot tracking evidence')
    report[key]={'frames':len(values),'mean_iou':float(np.mean(values)),'min_iou':min(values),
                 'iou_ge_08':sum(v>=.8 for v in values)}
(T/'tracking_quality.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
