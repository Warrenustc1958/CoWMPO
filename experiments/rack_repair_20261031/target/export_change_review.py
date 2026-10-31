# Copyright 2026 The RLinf Authors.
"""Export a review patch without changing the deployed repository or registry."""
import difflib,hashlib,json,subprocess
from pathlib import Path
FORMAL=Path('/mnt/nfs/fjj/Code/CoWMPO');SC=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v10');OUT=Path('/data_cf/cowmpo/staging/codex_s56_20261030_1020');parts=[];rows=[]
paths=list((SC/'toolkits/cowm').rglob('*.py'))+list((SC/'toolkits/cowm').rglob('*.sh'))+[SC/'rlinf/envs/sim/world_model/backend/code.py']
for p in sorted(paths):
    rel=p.relative_to(SC);q=FORMAL/rel
    if q.exists() and q.read_bytes()==p.read_bytes():continue
    before=q.read_text().splitlines(keepends=True) if q.exists() else [];after=p.read_text().splitlines(keepends=True)
    parts.extend(difflib.unified_diff(before,after,fromfile='a/'+str(rel) if q.exists() else '/dev/null',tofile='b/'+str(rel)))
    rows.append({'path':str(rel),'status':'modified' if q.exists() else 'new','sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
patch=OUT/'review_delta_v10.patch';patch.write_text(''.join(parts));manifest={'deployed_head':subprocess.check_output(['git','-C',str(FORMAL),'rev-parse','HEAD'],text=True).strip(),'deployed_status':subprocess.check_output(['git','-C',str(FORMAL),'status','--porcelain'],text=True),'staging':str(SC),'files':rows,'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'scope':'SDK and backend source delta only; experimental root scripts and world artifacts retained separately; no deployment or registry mutation'};(OUT/'review_delta_v10.json').write_text(json.dumps(manifest,indent=1));print(json.dumps({'files':len(rows),'paths':[r['path'] for r in rows],'deployed_status':manifest['deployed_status'],'patch':str(patch)},indent=1))
