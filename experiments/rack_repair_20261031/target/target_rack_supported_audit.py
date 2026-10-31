# Copyright 2026 The RLinf Authors.
"""Target-only six-dev audit of the rack candidate selected before reading GT."""
import json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');selection=json.loads((R/'rack_supported_response_dev_v2/selection.json').read_text());best=selection['best'];root=R/'rack_supported_response_dev_v2'/('candidate%02d'%best['seed']);rows=[]
for p in sorted(root.glob('*/state.json')):
    ep=p.parent.name;s=json.loads(p.read_text())['objects']['rack'];gt=json.loads((Path('/data_cf/cowmpo/target_eval/s5_gt/dev3')/(ep+'.json')).read_text())['rack']
    with np.load(p.parent/'render_masks.npz') as a:pred=a['rack']
    with np.load(Path('/data_cf/cowmpo/target_eval/s5_gt/dev3')/(ep+'_masks.npz')) as a:mask=a['rack']
    rows.append({'episode':ep,'xy_cm':float(np.linalg.norm(np.asarray(s['aabb_center_rel_base'][:2])-gt['aabb_center_rel_base'][:2])*100),'extent_cm':float(np.max(np.abs(np.asarray(s['aabb_half'])-gt['aabb_half']))*200),'render_gt_iou':float((pred&mask).sum()/(pred|mask).sum())})
report={'selection':best,'episodes':len(rows),'mean_xy_cm':float(np.mean([r['xy_cm'] for r in rows])),'p90_xy_cm':float(np.percentile([r['xy_cm'] for r in rows],90)),'within_3cm':sum(r['xy_cm']<3 for r in rows),'mean_max_extent_cm':float(np.mean([r['extent_cm'] for r in rows])),'mean_render_gt_iou':float(np.mean([r['render_gt_iou'] for r in rows])),'rows':rows,'scope':'six adaptive dev only; original conservative AABB metric retained'}
(T/'rack_supported_response_v2_diagnostic.json').write_text(json.dumps(report,indent=1));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=1))
