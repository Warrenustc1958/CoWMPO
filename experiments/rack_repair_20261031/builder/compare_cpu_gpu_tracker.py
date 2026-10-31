import json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');cpu=R/'grasp_video_cpu_pilot_dev_v1';gpu=R/'grasp_video_dev_v1/grasp';rows=[];same=True
for p in sorted(cpu.glob('*/association.json')):
    old=gpu/p.parent.name;ca=json.loads(p.read_text());ga=json.loads((old/'association.json').read_text())
    with np.load(p.parent/'masks.npz') as a,np.load(old/'masks.npz') as b:
        for n in ['robot','cream_cheese']:
            same &= ca['objects'][n]['status']==ga['objects'][n]['status']
            m=a[n];g=b[n];u=(m|g).sum();rows.append(float((m&g).sum()/u) if u else 1.)
report={'objects':len(rows),'identical_statuses':same,'mean_iou':float(np.mean(rows)),'min_iou':min(rows),'scope':'CPU versus prior GPU predictions, not independent GT quality validation'}
(R/'cpu_gpu_tracker_pilot_comparison.json').write_text(json.dumps(report,indent=1));print(json.dumps(report),flush=True)
if not same or min(rows)<.98:raise ValueError('CPU pilot differs materially from frozen GPU inference')
