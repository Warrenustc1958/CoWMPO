import json
from pathlib import Path

root=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/paired_raw_dev_v1')
for p in sorted(root.glob('*/*/recheck.json')):
    for n,e in json.loads(p.read_text())['log'].items():
        if e['status_before']!='ambiguous':continue
        print(p.parent.name,n,[(x['prompt'],x['scores']) for x in e.get('paired_scores',[]) if x['obj']==n])
