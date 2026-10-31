"""Bounded SAM2 point proposals for visually missing identities, using only RGB and a frozen bank."""
import hashlib
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

from cowm import roles
from cowm.init.associate import associate, mask_iou
from cowm.init.perceive import filter_reason
from cowm.init.recheck import tiles
from cowm.init.visual_identity import support
from extract_visual_features import descriptors
from visual_bank_analysis import classify

parser=argparse.ArgumentParser()
for key in ['source','out','bank','evidence','pool','data-role']:
    parser.add_argument('--'+key, required=True)
args=parser.parse_args()
SOURCE=Path(args.source)
OUTPUT=Path(args.out)
STRIDE=16
MAX_TILES=2
SAM_IOU=0.8
MAX_OBJECTS=3
if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':
    raise RuntimeError('GPU2 only')
if args.data_role not in ['dev','fit']:
    raise ValueError('dev/fit only')
roles.require([json.loads(p.read_text())['episode'] for p in SOURCE.glob('*/recheck.json')],
              {args.data_role}, 'bounded visual SAM proposals')
shutil.copytree(SOURCE,OUTPUT)
bank=np.load(args.bank, allow_pickle=False)
roles.require(bank['episodes'].tolist(), {'dev'}, 'visual bank')
model_dir=Path('/data_cf/cowmpo/builder_assets/models')
gp=AutoProcessor.from_pretrained(model_dir/'grounding-dino-base')
gm=AutoModelForZeroShotObjectDetection.from_pretrained(model_dir/'grounding-dino-base').cuda().eval()
sp=Sam2Processor.from_pretrained(model_dir/'sam2.1-hiera-large')
sm=Sam2Model.from_pretrained(model_dir/'sam2.1-hiera-large').cuda().eval()
prior=json.loads(Path('/data_cf/cowmpo/worlds/s5_v2/prior/perception_prior.json').read_text())
prior['robot_mask']=np.load('/data_cf/cowmpo/worlds/s5_v2/prior/perception_prior.npz')['robot_mask']
for path in sorted(SOURCE.glob('*/recheck.json')):
    rec=json.loads(path.read_text());ep=rec['episode'];dataset=path.parent.parent.name
    missing=sorted(n for n,r in rec['final'].items() if r['status']=='missing')[:MAX_OBJECTS]
    if not missing:
        continue
    roles.require([ep],{args.data_role},'visual SAM proposals')
    t0=time.perf_counter()
    pool=Path(args.pool)/ep
    audit=json.loads((pool/'audit.json').read_text())
    with np.load(pool/'candidate_masks.npz') as a:
        masks={c['cid']:a[f"c{c['cid']}"] for c in audit['candidates']}
    with np.load(path.parent/'recheck_masks.npz') as a:
        masks.update({int(k[1:]):a[k] for k in a.files})
    candidates=[c for c in audit['candidates'] if c['cid'] not in rec['removed']]+rec['added']
    image=Image.open(Path(args.evidence)/ep/'frame0.png').convert('RGB')
    W,H=image.size;occupied=np.zeros((H,W),bool)
    for r in rec['final'].values():
        if r['status']=='matched':
            occupied |= masks[r['cid']]
    regions=sorted(tiles(W,H),key=lambda t:(-float(1-occupied[t[1]:t[3],t[0]:t[2]].mean()),t))[:MAX_TILES]
    points=sorted(set((x,y) for x0,y0,x1,y1 in regions for y in range(y0+STRIDE//2,y1,STRIDE)
                      for x in range(x0+STRIDE//2,x1,STRIDE)))
    inputs=sp(images=image,input_points=[[[list(p)] for p in points]],
              input_labels=[[[1] for _ in points]],return_tensors='pt').to('cuda')
    with torch.no_grad():
        output=sm(**inputs,multimask_output=True)
    masks_out=sp.post_process_masks(output.pred_masks.cpu(),inputs['original_sizes'].cpu())[0].numpy().astype(bool)
    qualities=output.iou_scores[0].cpu().numpy()
    proposals=[]
    for pi,point in enumerate(points):
        for mi in range(masks_out.shape[1]):
            mk=masks_out[pi,mi];quality=float(qualities[pi,mi])
            if quality<SAM_IOU or not mk[point[1],point[0]] or not mk.any():
                continue
            ys,xs=np.nonzero(mk)
            box=[float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)]
            if all(filter_reason(box,n,prior) is not None for n in missing):
                continue
            proposals.append({'mask':mk,'quality':quality,'box':box,'point':point})
    if not proposals:
        print(ep,'NO_SAM_PROPOSALS',flush=True)
        continue
    inp=gp(images=image.resize((W*3,H*3)),text='object.',return_tensors='pt').to('cuda')
    with torch.no_grad():
        feature,_=gm.model.backbone(inp['pixel_values'],inp['pixel_mask'])
        vector=descriptors(feature,[p['mask'] for p in proposals])
    keep=bank['episodes']!=ep
    names=sorted(rec['final'])
    preds=classify(vector,bank['features'][keep],bank['labels'][keep].tolist(),names)
    accepted=[]
    for p,pred in zip(proposals,preds):
        n=pred['label']
        if n not in missing or support(n,pred)!='positive' or filter_reason(p['box'],n,prior) is not None:
            continue
        p.update(obj=n,prediction=pred,score=p['quality']*pred['gap'])
        accepted.append(p)
    chosen=[]
    for p in sorted(accepted,key=lambda p:(-p['score'],p['point'],p['box'])):
        if all(mask_iou(p['mask'],q['mask'])<0.9 for q in chosen):
            chosen.append(p)
    next_cid=max(masks,default=-1)+1
    for p in chosen:
        c={'cid':next_cid,'obj':p['obj'],'score':p['score'],'box':p['box'],'filtered':None,
           'source':'visual_sam_point','sam_iou_prediction':p['quality'],'visual_gap':p['prediction']['gap'],
           'point':list(p['point']),'prompt':'visual_bank'}
        candidates.append(c);rec['added'].append(c);masks[next_cid]=p['mask'];next_cid+=1
    final=associate(names,candidates,masks)
    rec['final']=final['objects']
    rec['sam_search']={'tiles':regions,'points':len(points),'masks_after_prior':len(proposals),
                       'positive_masks':len(accepted),'deduplicated_masks':len(chosen),'max_tiles':MAX_TILES,
                       'stride':STRIDE,'min_sam_iou_prediction':SAM_IOU,'score':'sam_iou_prediction * visual_class_gap'}
    rec['seconds']=time.perf_counter()-t0;rec['crops']+=len(regions)
    for n in missing:
        rec['log'][n]={'status_before':'missing','status_after':rec['final'][n]['status'],
                       'cid_after':rec['final'][n]['cid'],'action':'bounded_visual_sam_search'}
    rec['protocol']='visual_Loo_sam_v2'
    dest=OUTPUT/ep
    (dest/'recheck.json').write_text(json.dumps(rec,indent=1))
    np.savez_compressed(dest/'recheck_masks.npz',**{f"c{c['cid']}":masks[c['cid']] for c in rec['added']})
    np.savez_compressed(dest/'masks.npz',**{n:masks[r['cid']] for n,r in rec['final'].items() if r['cid'] is not None})
    association=json.loads((dest/'association.json').read_text())
    association['objects']=rec['final'];association['algorithm']='frozen_visual_sam_v1'
    association['masks_sha256']=hashlib.sha256((dest/'masks.npz').read_bytes()).hexdigest()
    association['sam_search']=rec['sam_search']
    (dest/'association.json').write_text(json.dumps(association,indent=1))
    print(json.dumps({'role':args.data_role,'searched_objects':len(missing),'proposals':rec['sam_search'],
                      'status_counts':{s:sum(rec['final'][n]['status']==s for n in missing)
                                       for s in ['matched','ambiguous','missing']},'seconds':rec['seconds']}),flush=True)
