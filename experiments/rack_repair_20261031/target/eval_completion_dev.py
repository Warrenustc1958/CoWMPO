# Copyright 2026 The RLinf Authors.
"""Target-only dev completeness check; reports no geometry to Builder."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from cowm import roles
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
stats=defaultdict(list);changes=[]
def iou(a,b):
    u=(a|b).sum();return float((a&b).sum()/u) if u else 0.
for p in sorted((R/'complete_dev_masks_v1').glob('*/masks.npz')):
    ep=p.parent.name;roles.require([ep],{'dev'},'Target completion diagnosis')
    dataset='dev3' if (R/'visual_dev_bound_v1/dev3'/ep).exists() else 'fit3'
    with np.load(Path('/data_cf/cowmpo/target_eval/s5_gt')/dataset/(ep+'_masks.npz')) as a:gt={n:a[n] for n in a.files}
    with np.load(R/'visual_dev_bound_v1'/dataset/ep/'masks.npz') as a:before={n:a[n] for n in a.files}
    with np.load(p) as a:after={n:a[n] for n in a.files}
    for n,mk in after.items():
        b=iou(before[n],gt[n]);v=iou(mk,gt[n]);best=max(gt,key=lambda k:iou(mk,gt[k]))
        stats[n].append({'before':b,'after':v,'identity_correct':bool(best==n and v>=.3)})
        if not np.array_equal(mk,before[n]):changes.append({'episode':ep,'object':n,'before':b,'after':v})
summary={n:{'instances':len(rows),'before_iou':float(np.mean([r['before'] for r in rows])),
 'after_iou':float(np.mean([r['after'] for r in rows])),'wrong_identity':sum(not r['identity_correct'] for r in rows),
 'quality_decreased_gt_005':sum(r['after']<r['before']-.05 for r in rows)} for n,rows in stats.items()}
(T/'completion_dev_v1.json').write_text(json.dumps({'summary':summary,'changes_dev':changes},indent=1))
print(json.dumps(summary,indent=1))
