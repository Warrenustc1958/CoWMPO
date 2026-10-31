"""Target-side summaries of completed development diagnostics."""
import json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
d=json.loads((R/'rack_touch_fit_dev_v1/selection.json').read_text())
print('RACK_TOP', [{k:c[k] for k in ('index','family','dx','static_iou','robot_iou','loss')} for c in sorted(d['candidates'],key=lambda c:c['loss'])[:6]])
ep='libero_goal_t0_i0'
gt=json.loads((Path('/data_cf/cowmpo/target_eval/s5_gt/dev3')/(ep+'.json')).read_text())['rack']
rows=[]
for tag,src in [('original',R/'stove_parts_states_dev_v1'/ep),('selected',Path(d['best']['state']).parent)]:
    ob=json.loads((src/'state.json').read_text())['objects']['rack']
    rows.append({'version':tag,'xy_cm':float(np.linalg.norm(np.asarray(ob['aabb_center_rel_base'][:2])-gt['aabb_center_rel_base'][:2])*100),
      'extent_max_cm':float(np.max(np.abs(np.asarray(ob['aabb_half'])-gt['aabb_half']))*200)})
report={'episodes':1,'rack':rows,'scope':'Target-only diagnostic of RGB/action-selected candidate; not validation'}
(T/'rack_touch_fit_diagnostic_v1.json').write_text(json.dumps(report,indent=1))
print('RACK_DIAGNOSTIC',report)
for name in ['grasp_contrast_dev_v1','grasp_contrast_dev_holdout_v1']:
    results=[]
    for p in sorted((T/name).glob('*.json')):
        d=json.loads(p.read_text())
        if 'label' in d:results.append({k:d[k] for k in ('key','label','prediction','lift_cm','finger_bodies')})
    print(name,results)
