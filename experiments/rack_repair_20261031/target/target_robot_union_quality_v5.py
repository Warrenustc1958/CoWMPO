# Copyright 2026 The RLinf Authors.
"""Private comparison of robot tracking with exact frame and anchor preservation."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');SRC=R/'rack_robot_union_v5';OUT=T/'robot_union_quality_v5.json';EP='libero_goal_t0_i0'
frozen=json.loads((SRC/'freeze.json').read_text());complete=json.loads((SRC/'complete.json').read_text())
if {r['key'] for r in complete}!={r['key'] for r in frozen['queries']}:raise ValueError('incomplete query set')
rows=[]
for rec in frozen['queries']:
    oldroot=R/('rack_isolated_video_cpu_v2' if rec['group']=='horizontal' else 'rack_vertical_video_cpu_v3')/rec['key'];gtroot=T/('rack_isolated_dev_v2' if rec['group']=='horizontal' else 'rack_vertical_dev_v3')/rec['key'];values=[];old=[];statuses=[]
    expected={f'{EP}__f{s:03d}' for s in rec['steps']};newroot=SRC/'frames'/rec['key']
    if {p.parent.name for p in newroot.glob('*/association.json')}!=expected:raise ValueError('frame count changed')
    for step in rec['steps']:
        frame=f'{EP}__f{step:03d}'
        with np.load(gtroot/frame/'gt_masks.npz') as a:gt=a['robot']
        with np.load(oldroot/frame/'masks.npz') as a:base=a['robot']
        with np.load(newroot/frame/'masks.npz') as a:new=a['robot']
        meta=json.loads((newroot/frame/'association.json').read_text())
        if meta['masks_sha256']!=hashlib.sha256((newroot/frame/'masks.npz').read_bytes()).hexdigest():raise ValueError('new masks changed')
        if step==0 and not np.array_equal(base,new):raise ValueError('initial anchor changed')
        old.append(float((gt&base).sum()/(gt|base).sum()));values.append(float((gt&new).sum()/(gt|new).sum()));statuses.append(meta['objects']['robot']['status'])
    rows.append({'key':rec['key'],'frames':len(values),'old_mean_iou':float(np.mean(old)),'new_mean_iou':float(np.mean(values)),'old_min_iou':min(old),'new_min_iou':min(values),'new_iou_ge_05':sum(v>=.5 for v in values),'new_iou_ge_08':sum(v>=.8 for v in values),'matched':statuses.count('matched'),'per_frame_iou':values})
report={'scope':'Target-private dev quality audit; no GT-derived prompts; exact same initial anchor and RGB','queries':rows}
OUT.write_text(json.dumps(report,indent=1));print(json.dumps({'queries':[{k:v for k,v in r.items() if k!='per_frame_iou'} for r in rows]},indent=1))
