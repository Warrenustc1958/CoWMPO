"""Apply leave-one-episode-out visual evidence to the frozen dev pool; no GT."""
import json
import os
from pathlib import Path

import numpy as np

from cowm import roles
from cowm.init.perceive import filter_reason
from cowm.init.visual_identity import semantic_associate

ROOT=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
features=ROOT/'visual_dev_v1'
analysis=ROOT/'visual_dev_analysis_v1'
output=ROOT/'visual_semantic_dev_v1'
output.mkdir(exist_ok=False)
prior=json.loads(Path('/data_cf/cowmpo/worlds/s5_v2/prior/perception_prior.json').read_text())
prior['robot_mask']=np.load('/data_cf/cowmpo/worlds/s5_v2/prior/perception_prior.npz')['robot_mask']
for path in sorted(features.glob('*/*.json')):
    meta=json.loads(path.read_text());ep=meta['episode'];dataset=path.parent.name
    roles.require([ep],{'dev'},'visual development')
    pred=json.loads((analysis/(ep+'_visual.json')).read_text())
    predictions=dict(zip(pred['cids'],pred['predictions']))
    source=Path('/data_cf/cowmpo/worlds/s5_audit_v1')/dataset/ep
    audit=json.loads((source/'audit.json').read_text())
    with np.load(source/'candidate_masks.npz') as a:
        masks={c['cid']:a[f"c{c['cid']}"] for c in audit['candidates']}
    names=list(meta['base'])
    result=semantic_associate(names,audit['candidates'],masks,predictions,
                       prior_filter=lambda b,n:filter_reason(b,n,prior))
    dest=output/dataset/ep;dest.mkdir(parents=True)
    log={n:{'status_before':meta['base'][n]['status'],'status_after':r['status'],'cid_after':r['cid']}
         for n,r in result['objects'].items() if (r['status'],r['cid'])!=(meta['base'][n]['status'],meta['base'][n]['cid'])}
    record={'episode':ep,'base':meta['base'],'final':result['objects'],'log':log,
            'removed':result['rejected'],'added':result['added'],'crops':0,'dino_forward_passes':0,
            'seconds':0,'changed_untriggered':[],'protocol':'visual_semantic_Loo_v1'}
    (dest/'recheck.json').write_text(json.dumps(record,indent=1))
    np.savez_compressed(dest/'recheck_masks.npz',**{f"c{c['cid']}":result['masks'][c['cid']] for c in result['added']})
    print(ep,{n:(meta['base'][n]['status'],r['status']) for n,r in result['objects'].items()
              if meta['base'][n]['status']!=r['status']},'shared',len(result['added']))
