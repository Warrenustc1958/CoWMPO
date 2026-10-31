# Copyright 2026 The RLinf Authors.
"""Bounded paired DINO verification in one shared crop; only unresolved objects can change."""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from cowm import roles
from cowm.init.associate import associate, mask_overlap
from cowm.init.paired_identity import preference
from cowm.init.paired_recheck import paired_scores
from cowm.init.recheck import crop_around, to_original

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=argparse.ArgumentParser()
for key in ['source','out','pool','evidence','world','freeze','data-role']:
    parser.add_argument('--'+key,required=True)
args=parser.parse_args()
if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise RuntimeError('GPU2 only')
if args.data_role not in ['dev','fit']:raise ValueError('dev/fit only')
frozen=json.loads(Path(args.freeze).read_text())
for p,digest in frozen['files'].items():
    if sha(p)!=digest:raise ValueError('frozen file changed: '+p)
paths=sorted(Path(args.source).glob('*/recheck.json'))
roles.require([json.loads(p.read_text())['episode'] for p in paths],{args.data_role},'paired verification')
shutil.copytree(args.source,args.out)
world=json.loads((Path(args.world)/'world.json').read_text())
objects={o['name']:o for o in world['objects']}
model='/data_cf/cowmpo/builder_assets/models/grounding-dino-base'
proc=AutoProcessor.from_pretrained(model)
net=AutoModelForZeroShotObjectDetection.from_pretrained(model).cuda().eval()
totals={'episodes':len(paths),'triggered':0,'resolved':0,'crops':0,'forwards':0,'seconds':0.}
for path in paths:
    rec=json.loads(path.read_text());ep=rec['episode']
    pending=sorted(n for n,r in rec['final'].items() if r['status']=='ambiguous')[:3]
    if not pending:continue
    t0=time.perf_counter()
    pool=Path(args.pool)/ep
    audit=json.loads((pool/'audit.json').read_text())
    with np.load(pool/'candidate_masks.npz') as a:masks={int(k[1:]):a[k] for k in a.files}
    with np.load(path.parent/'recheck_masks.npz') as a:masks.update({int(k[1:]):a[k] for k in a.files})
    candidates=[c for c in audit['candidates'] if c['cid'] not in rec['removed']]+rec['added']
    by_cid={c['cid']:c for c in candidates}
    image=Image.open(Path(args.evidence)/ep/'frame0.png').convert('RGB');W,H=image.size
    paired={}
    for name in pending:
        r=rec['final'][name];totals['triggered']+=1
        if r['alt_cid'] is None:continue
        regions={k:by_cid[cid]['box'] for k,cid in [('chosen',r['cid']),('alt',r['alt_cid'])]}
        boxes=list(regions.values())
        union=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
        crop=crop_around(union,W,H);sub=image.crop(crop)
        scale=768/max(sub.size);big=sub.resize(tuple(max(1,round(s*scale)) for s in sub.size),Image.BICUBIC)
        obj=objects[name];prompts=sorted({obj['name'].replace('_',' '),obj['category'].replace('_',' ')})
        dets=[];totals['crops']+=1
        for prompt in prompts:
            inputs=proc(images=big,text=prompt+'.',return_tensors='pt').to('cuda')
            with torch.no_grad():o=net(**inputs)
            totals['forwards']+=1
            result=proc.post_process_grounded_object_detection(o,inputs.input_ids,threshold=0.,text_threshold=0.,
                     target_sizes=[(sub.size[1],sub.size[0])])[0]
            dets += [{'obj':name,'prompt':prompt,'score':score,'box':to_original(box,crop)}
                     for score,box in zip(result['scores'].tolist(),result['boxes'].tolist())]
        rows=paired_scores(regions,dets);decision=preference(rows,prompts)
        paired[name]={'crop':crop,'prompts':prompts,'scores':rows,'decision':decision}
        if decision is None:continue
        winner=r['cid'] if decision=='chosen' else r['alt_cid']
        reduced=[c for c in candidates if c['obj']!=name or mask_overlap(masks[c['cid']],masks[winner])>=.7]
        trial=associate(sorted(rec['final']),reduced,masks)['objects']
        if trial[name]['status']!='matched':continue
        if any((trial[n]['status'],trial[n]['cid'])!=(rec['final'][n]['status'],rec['final'][n]['cid'])
               for n in trial if n!=name):continue
        candidates=reduced;rec['final']=trial;totals['resolved']+=1
    elapsed=time.perf_counter()-t0;totals['seconds']+=elapsed
    rec['paired_verification']=paired;rec['paired_seconds']=elapsed
    dest=Path(args.out)/ep
    (dest/'recheck.json').write_text(json.dumps(rec,indent=1))
    np.savez_compressed(dest/'masks.npz',**{n:masks[r['cid']] for n,r in rec['final'].items() if r['cid'] is not None})
    bound=json.loads((dest/'association.json').read_text());bound['objects']=rec['final']
    bound['masks_sha256']=sha(dest/'masks.npz');bound['paired_freeze_sha256']=sha(args.freeze)
    bound['algorithm']='frozen_visual_paired_v3'
    (dest/'association.json').write_text(json.dumps(bound,indent=1))
(Path(args.out)/'paired_summary.json').write_text(json.dumps(totals,indent=2))
print(json.dumps(totals))
