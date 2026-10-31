"""Target-only trajectory comparison; do not publish privileged state samples."""
import json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
rows=[]
for p in sorted((R/'contact_response_dev_v1').glob('*/eef.npy')):
    key=p.parent.name;pred=np.load(p);gt=np.load(T/'contact_diagnosis_dev_v1'/(key+'_eef.npy'))
    if pred.shape!=gt.shape:raise ValueError('trajectory length mismatch')
    err=np.linalg.norm(pred-gt,axis=1)*1000
    rows.append({'key':key,'eef_error_mm':{str(i+1):float(err[i]) for i in [59,119,129,159,199,219,239] if i<len(err)},'max_mm':float(err.max())})
(T/'contact_trajectory_comparison_v1.json').write_text(json.dumps(rows,indent=1));print(json.dumps(rows,indent=1))
