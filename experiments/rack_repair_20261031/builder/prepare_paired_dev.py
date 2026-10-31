import json
import hashlib
import shutil
from pathlib import Path
import numpy as np
from cowm.init.paired_identity import preference

R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
source=R/'paired_raw_dev_v1';out=R/'paired_dev_source_v3'
shutil.copytree(source,out)
count=accepted=0
for path in out.glob('*/*/recheck.json'):
    rec=json.loads(path.read_text());ep=rec['episode'];dataset=path.parent.parent.name
    for n,entry in rec['log'].items():
        if entry['status_before']=='ambiguous':
            rows=[r for r in entry['paired_scores'] if r['obj']==n]
            prompts=[n.replace('_',' '),'box'] if n=='cream_cheese' else [n.replace('_',' ')]
            accepted+=preference(rows,prompts)=='chosen';count+=1
    with np.load(Path('/data_cf/cowmpo/worlds/s5_audit_v1')/dataset/ep/'candidate_masks.npz') as a:
        masks={int(k[1:]):a[k] for k in a.files}
    np.savez_compressed(path.parent/'masks.npz',**{n:masks[r['cid']] for n,r in rec['final'].items() if r['cid'] is not None})
    (path.parent/'association.json').write_text(json.dumps({'schema':1,'episode':ep,'objects':rec['final'],
        'masks_sha256':hashlib.sha256((path.parent/'masks.npz').read_bytes()).hexdigest()}))
print({'cached_dev_comparisons':count,'accepted_chosen':accepted})
