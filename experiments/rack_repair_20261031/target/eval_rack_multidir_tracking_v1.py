# Copyright 2026 The RLinf Authors.
"""Target-only quality diagnostic for the shared-robot video evidence."""
import json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/rack_multidir_video_cpu_v1')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/rack_multidir_dev_v1')
report={}
keys=json.loads((Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/rack_multidir_preflight_dev_v1')/'selection.json').read_text())['selected']
for key in keys:
    values=[]
    for path in sorted((R/key).glob('*/association.json')):
        with np.load(path.parent/'masks.npz') as a:pred=a['robot']
        association=json.loads(path.read_text())
        if association['objects']['robot']['status']!='matched':raise ValueError('unmatched tracking evidence')
        with np.load(T/key/path.parent.name/'gt_masks.npz') as a:gt=a['robot']
        values.append(float((pred&gt).sum()/(pred|gt).sum()))
    if len(values)!=36:raise ValueError('incomplete robot tracking evidence')
    report[key]={'frames':len(values),'mean_iou':float(np.mean(values)),'min_iou':min(values),
                 'iou_ge_08':sum(v>=.8 for v in values)}
(T/'tracking_quality.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
