# Copyright 2026 The RLinf Authors.
"""Dev-only whole-object candidate completion; no new model calls or GT inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from cowm import roles
from cowm.init.visual_identity import support

R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'complete_dev_masks_v1';OUT.mkdir(exist_ok=False)
robot=np.load('/data_cf/cowmpo/worlds/s5_v2/prior/perception_prior.npz')['robot_mask']
records=[]
for source in sorted((R/'visual_dev_bound_v1').glob('*/*/association.json')):
    ep=source.parent.name;dataset=source.parent.parent.name
    roles.require([ep],{'dev'},'whole-object completion development')
    audit_root=Path('/data_cf/cowmpo/worlds/s5_audit_v1')/dataset/ep
    audit=json.loads((audit_root/'audit.json').read_text())
    pred=json.loads((R/'visual_dev_analysis_v1'/(ep+'_visual.json')).read_text())
    preds=dict(zip(pred['cids'],pred['predictions']))
    rec=json.loads(source.read_text())
    with np.load(source.parent/'masks.npz') as a:masks={n:a[n] for n in a.files}
    with np.load(audit_root/'candidate_masks.npz') as a:pool={int(k[1:]):a[k] for k in a.files}
    changes={}
    for name,base in list(masks.items()):
        others=robot.copy()
        for n,mk in masks.items():
            if n!=name:others|=mk
        eligible=[]
        for c in audit['candidates']:
            if c['obj']!=name or c['filtered'] is not None or c['score']<.2:continue
            mk=pool[c['cid']];area=mk.sum();ratio=area/max(base.sum(),1)
            if not (1.05<=ratio<=1.6):continue
            if (mk&base).sum()/max(base.sum(),1)<.95:continue
            if (mk&others).sum()/max(area,1)>.1:continue
            if support(name,preds[c['cid']])=='negative':continue
            eligible.append(c)
        if eligible:
            pick=max(eligible,key=lambda c:(int(pool[c['cid']].sum()),c['score'],-c['cid']))
            masks[name]=pool[pick['cid']]
            changes[name]={'before_area':int(base.sum()),'after_area':int(masks[name].sum()),'cid':pick['cid']}
            rec['objects'][name]['cid']=pick['cid']
            rec['objects'][name]['completion']=changes[name]
    dest=OUT/ep;dest.mkdir();np.savez_compressed(dest/'masks.npz',**masks)
    rec['algorithm']='visual_v2_with_dev_completion_v1'
    rec['source_association_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    rec['masks_sha256']=hashlib.sha256((dest/'masks.npz').read_bytes()).hexdigest()
    (dest/'association.json').write_text(json.dumps(rec,indent=1))
    records.append({'episode':ep,'changes':changes})
(OUT/'completion.json').write_text(json.dumps(records,indent=1))
print(json.dumps(records,indent=1))
