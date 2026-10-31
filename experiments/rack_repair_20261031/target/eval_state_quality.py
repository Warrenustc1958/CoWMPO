# Copyright 2026 The RLinf Authors.
"""Target-only aggregate state quality, including localization tails and mask quality."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from cowm import roles
p=argparse.ArgumentParser();p.add_argument('--est',required=True);p.add_argument('--out',required=True)
p.add_argument('--role',choices=['dev','fit'],required=True);a=p.parse_args()
stats=defaultdict(list)
def iou(x,y):
    total=(x|y).sum();return float((x&y).sum()/total) if total else 0.
for path in sorted(Path(a.est).glob('*/state.json')):
    ep=path.parent.name;roles.require([ep],{a.role},'Target aggregate state evaluation')
    data='dev3' if roles.parse(ep)[1] in [0,1,2] else 'fit3'
    gpath=Path('/data_cf/cowmpo/target_eval/s5_gt')/data
    st=json.loads(path.read_text())['objects'];gt=json.loads((gpath/(ep+'.json')).read_text())
    with np.load(gpath/(ep+'_masks.npz')) as x:gm={n:x[n] for n in x.files}
    with np.load(path.parent/'masks.npz') as x:sm={n:x[n] for n in x.files}
    with np.load(path.parent/'render_masks.npz') as x:rm={n:x[n] for n in x.files}
    for n,g in gt.items():
        err=np.asarray(st[n]['aabb_center_rel_base'])-np.asarray(g['aabb_center_rel_base'])
        size=np.abs(np.asarray(st[n]['aabb_half'])-np.asarray(g['aabb_half']))*200
        stats[n].append({'xy_cm':float(np.linalg.norm(err[:2])*100),'z_cm':float(abs(err[2])*100),
            'extent_cm':float(size.max()),'mask_iou':iou(sm[n],gm[n]),'render_iou':iou(rm[n],gm[n])})
report={}
for n,rows in stats.items():
    report[n]={'instances':len(rows),'within_3cm':sum(r['xy_cm']<3 for r in rows),
      'over_10cm':sum(r['xy_cm']>10 for r in rows)}
    for key in rows[0]:
        vals=[r[key] for r in rows];report[n][key+'_mean']=float(np.mean(vals));report[n][key+'_p90']=float(np.percentile(vals,90))
report['_all']={'instances':sum(len(r) for r in stats.values()),
 'within_3cm':sum(r['xy_cm']<3 for rows in stats.values() for r in rows),
 'role':a.role,'scope':'aggregate diagnostic only; no GT input to Builder'}
Path(a.out).write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
