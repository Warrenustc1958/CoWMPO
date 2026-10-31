# Copyright 2026 The RLinf Authors.
"""Combine whole-robot and part memories without episode-specific selection."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SOURCE=R/'rack_robot_parts_cpu_v4';OUT=R/'rack_robot_union_v5';EP='libero_goal_t0_i0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen=json.loads((SOURCE/'freeze.json').read_text());complete=json.loads((SOURCE/'complete.json').read_text())
if {r['key'] for r in complete}!={r['key'] for r in frozen['queries']}:raise ValueError('incomplete parts source')
OUT.mkdir(exist_ok=False);files={}
for rec in frozen['queries']:
    old=R/('rack_isolated_video_cpu_v2' if rec['group']=='horizontal' else 'rack_vertical_video_cpu_v3')/rec['key']
    for step in rec['steps']:
        for root in [old,SOURCE/'frames'/rec['key']]:
            for name in ['association.json','masks.npz']:
                p=root/f'{EP}__f{step:03d}'/name;files[str(p)]=sha(p)
(OUT/'freeze.json').write_text(json.dumps({'queries':frozen['queries'],'files':files,'script_sha256':sha(__file__),
    'rule':'union positive-presence whole robot mask with independent shared-platform part masks on every frame; no GT thresholds or per-episode choice; post hoc dev repair','GT_inputs':False},indent=1))
for rec in frozen['queries']:
    old=R/('rack_isolated_video_cpu_v2' if rec['group']=='horizontal' else 'rack_vertical_video_cpu_v3')/rec['key']
    for step in rec['steps']:
        frame=f'{EP}__f{step:03d}';inputs=[old/frame,SOURCE/'frames'/rec['key']/frame];union=np.zeros((256,256),bool);sources=[];areas=[]
        for src in inputs:
            meta=json.loads((src/'association.json').read_text())
            if meta['masks_sha256']!=sha(src/'masks.npz'):raise ValueError('mask binding changed')
            with np.load(src/'masks.npz') as a:mask=a['robot']
            if meta['objects']['robot']['status']=='matched':union|=mask
            areas.append(int(mask.sum()));sources.append({'path':str(src),'association_sha256':sha(src/'association.json')})
        dest=OUT/'frames'/rec['key']/frame;dest.mkdir(parents=True);np.savez_compressed(dest/'masks.npz',robot=union)
        (dest/'association.json').write_text(json.dumps({'episode':frame,'algorithm':'robot_whole_parts_union_v5','objects':{'robot':{'status':'matched' if union.any() else 'missing','identity_source':'whole and shared-platform part memory union','input_areas':areas,'union_area':int(union.sum())}},'masks_sha256':sha(dest/'masks.npz'),'sources':sources},indent=1))
for p,h in files.items():
    if sha(p)!=h:raise ValueError('input changed')
(OUT/'complete.json').write_text(json.dumps(complete,indent=1));print('ROBOT_UNION_DONE',sum(r['frames'] for r in complete))
