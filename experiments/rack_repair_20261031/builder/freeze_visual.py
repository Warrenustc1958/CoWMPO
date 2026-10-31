# Copyright 2026 The RLinf Authors.
"""Freeze dev-only visual bank, rules, code and model bytes before fit validation."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from cowm import roles

ROOT=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
SC=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
bank=ROOT/'visual_dev_analysis_v1/report.npz'
b=np.load(bank,allow_pickle=False)
roles.require(b['episodes'].tolist(), {'dev'}, 'freeze visual bank')
paths=[bank, SC/'extract_visual_features.py', SC/'visual_bank_analysis.py',
       SC/'run_visual_association.py',SC/'propose_missing_visual.py',
       SC/'toolkits/cowm/init/visual_identity.py', SC/'toolkits/cowm/init/associate.py',
       SC/'toolkits/cowm/init/perceive.py', SC/'toolkits/cowm/profiles/splits_goal_v1.json',
       Path('/data_cf/cowmpo/worlds/s5_v2/prior/perception_prior.json'),
       Path('/data_cf/cowmpo/worlds/s5_v2/prior/perception_prior.npz')]
model_paths=[]
for model in ['grounding-dino-base','sam2.1-hiera-large']:
    model_paths += sorted(p for p in (Path('/data_cf/cowmpo/builder_assets/models')/model).rglob('*') if p.is_file())
paths += model_paths
record={'schema':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'bank_sha256':sha(bank),'bank_labels':'v3 matched dev pseudo-labels; no GT',
        'bank_examples':len(b['labels']), 'bank_episodes':sorted(set(b['episodes'].tolist())),
        'validation':'dev leave-one-episode-out; fit full frozen dev bank',
        'development':'posthoc on dev; not independent validation',
        'rules':{'visual_margin':.05,'neighbors':3,'mask_pooling':'per-scale norm, then concatenate and normalize',
                 'sam_stride':16,'sam_max_tiles':2,'sam_max_missing_objects':3,'sam_iou_prediction_min':.8},
        'files':{str(p):sha(p) for p in paths}}
out=ROOT/'visual_freeze_v1.json'
with out.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({'freeze':str(out),'sha256':sha(out),'bank_examples':len(b['labels']),
                  'bank_episodes':len(set(b['episodes'].tolist())),'frozen_files':len(paths)}))
