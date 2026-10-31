import datetime
import hashlib
import json
from pathlib import Path
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
SC=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4')
record=json.loads((R/'visual_freeze_v3_paired.json').read_text())
for p,digest in record['files'].items():
    if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest:raise ValueError('frozen source changed: '+p)
for p in [SC/'track_probe_video.py',SC/'toolkits/cowm/profiles/splits_goal_v1.json']:
    record['files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
record.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   video_protocol={'model':'SAM2VideoModel existing complete checkpoint; exact load required',
      'dtype':'float32','presence_logit_threshold':0.,'non_overlap':True,'frame0_original_anchor':True,
      'alignment':{'pixel_threshold':25,'global_changed_fraction_max':.005,'object_changed_fraction_max':.2}},
   validation_scope='developed/validated on dev only; video not yet run on fit')
with (R/'video_freeze_v1.json').open('x') as f:json.dump(record,f,indent=2)
print('VIDEO_FROZEN',hashlib.sha256((R/'video_freeze_v1.json').read_bytes()).hexdigest())
