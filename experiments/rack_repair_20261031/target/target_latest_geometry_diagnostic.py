"""Target-only diagnostics for candidates selected using RGB and action responses."""
import json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
report={}
selection=json.loads((R/'rack_touch_fit_dev_v2/selection.json').read_text());best=selection['best'];ep='libero_goal_t0_i0'
st=json.loads(Path(best['state']).read_text())['objects']['rack'];gt=json.loads((Path('/data_cf/cowmpo/target_eval/s5_gt/dev3')/(ep+'.json')).read_text())['rack']
report['rack']={'candidate':best['index'],'family':best['family'],'dx':best['dx'],'xy_cm':float(np.linalg.norm(np.asarray(st['aabb_center_rel_base'][:2])-gt['aabb_center_rel_base'][:2])*100),'extent_cm':float(np.max(np.abs(np.asarray(st['aabb_half'])-gt['aabb_half']))*200),'robot_iou':best['robot_iou']}
selection=json.loads((R/'bowl_support_dev_v1/selection.json').read_text());best=selection['best'];rows=[]
for p in sorted(Path(best['states']).glob('*/state.json')):
    ep=p.parent.name;st=json.loads(p.read_text())['objects']['bowl'];gt=json.loads((Path('/data_cf/cowmpo/target_eval/s5_gt/dev3')/(ep+'.json')).read_text())['bowl']
    rows.append({'xy_cm':float(np.linalg.norm(np.asarray(st['aabb_center_rel_base'][:2])-gt['aabb_center_rel_base'][:2])*100),'z_cm':float(abs(st['aabb_center_rel_base'][2]-gt['aabb_center_rel_base'][2])*100),'extent_cm':float(np.max(np.abs(np.asarray(st['aabb_half'])-gt['aabb_half']))*200)})
report['bowl']={'candidate':best['index'],'scales':best['scales'],'episodes':len(rows),'mean_mask_iou':best['mean_mask_iou'],'means':{k:float(np.mean([r[k] for r in rows])) for k in rows[0]}}
(T/'latest_geometry_diagnostic_v1.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
