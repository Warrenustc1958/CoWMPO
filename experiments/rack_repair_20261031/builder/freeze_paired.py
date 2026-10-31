import datetime
import hashlib
import json
from pathlib import Path
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
SC=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v2')
record=json.loads((R/'visual_freeze_v2.json').read_text())
for p in [SC/'paired_resolve.py',SC/'toolkits/cowm/init/paired_identity.py',
          SC/'toolkits/cowm/init/paired_recheck.py',SC/'toolkits/cowm/init/recheck.py']:
    record['files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
record.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              paired_rule={'prompts':'world literal name and category, at least two distinct prompts',
                           'margin':.05,'same_crop':True,'max_objects':3,'changes_to_other_objects':'forbidden'})
with (R/'visual_freeze_v3_paired.json').open('x') as f:json.dump(record,f,indent=2)
print('PAIRED_FROZEN')
