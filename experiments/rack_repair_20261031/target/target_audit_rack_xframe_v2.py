# Copyright 2026 The RLinf Authors.
"""Target-private physical geometry audit after public-evidence candidate selection."""
import json
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene,iou
from cowm.init.aabb import object_aabb
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
best=json.loads((R/'rack_xframe_multidir_candidates_dev_v2/selection.json').read_text())['best'];scene=Scene(best['world']);rows=[]
for k in range(6):
    ep=f'libero_goal_t{k}_i0';doc=json.loads((Path(best['states'])/ep/'state.json').read_text());scene.reset();apply_state(scene,doc['objects'],require_matched=True,robot=json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot'])
    index=scene.names.index('rack');m,d=scene.env.sim.model,scene.env.sim.data;c,h=object_aabb(m,d,np.nonzero(scene.geom2obj==index)[0]);gt=json.loads((Path('/data_cf/cowmpo/target_eval/s5_gt/dev3')/(ep+'.json')).read_text())['rack']
    with np.load(Path('/data_cf/cowmpo/target_eval/s5_gt/dev3')/(ep+'_masks.npz')) as a:mask=a['rack']
    rows.append({'episode':ep,'xy_cm':float(np.linalg.norm((c-scene.base)[:2]-gt['aabb_center_rel_base'][:2])*100),'extent_cm':float(np.max(np.abs(h-gt['aabb_half']))*200),'render_gt_iou':iou(scene.seg()==index,mask)})
scene.env.close();report={'selection':best,'episodes':6,'mean_xy_cm':float(np.mean([r['xy_cm'] for r in rows])),'within_3cm':sum(r['xy_cm']<3 for r in rows),'mean_max_extent_cm':float(np.mean([r['extent_cm'] for r in rows])),'mean_render_gt_iou':float(np.mean([r['render_gt_iou'] for r in rows])),'rows':rows,'scope':'six adaptive dev; old conservative AABB metric retained; GT not fed to Builder'}
(T/'rack_xframe_multidir_v2_diagnostic.json').write_text(json.dumps(report,indent=1));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=1))
