# Copyright 2026 The RLinf Authors.
"""Temporal RGB anchors plus frozen visual prototypes for existing probe-mask proposals; no GT."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
from cowm import roles
from cowm.init.visual_identity import reassociate
from cowm.init.perceive import prompts_for, filter_reason
from extract_visual_features import descriptors
from visual_bank_analysis import classify

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=argparse.ArgumentParser()
for key in ['old-masks','evidence','anchors','initial-evidence','bank','probes','out','data-role','world','prior']:
    parser.add_argument('--'+key,required=True)
args=parser.parse_args()
if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise RuntimeError('GPU2 only')
if args.data_role not in ['dev','fit']:raise ValueError('dev/fit only')
episodes=sorted(p.name for p in Path(args.anchors).iterdir() if p.is_dir())
roles.require(episodes,{args.data_role},'probe identity development/validation')
out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
bank=np.load(args.bank,allow_pickle=False)
roles.require(bank['episodes'].tolist(),{'dev'},'visual bank')
model='/data_cf/cowmpo/builder_assets/models/grounding-dino-base'
proc=AutoProcessor.from_pretrained(model)
net=AutoModelForZeroShotObjectDetection.from_pretrained(model).cuda().eval()
sam_path='/data_cf/cowmpo/builder_assets/models/sam2.1-hiera-large'
sam_proc=Sam2Processor.from_pretrained(sam_path)
sam=Sam2Model.from_pretrained(sam_path).cuda().eval()
world=json.loads((Path(args.world)/'world.json').read_text())
object_specs={o['name']:o for o in world['objects']}
prior=json.loads((Path(args.prior)/'perception_prior.json').read_text())
prior['robot_mask']=np.load(Path(args.prior)/'perception_prior.npz')['robot_mask']
totals={'episodes':len(episodes),'frames':0,'objects':0,'matched':0,'ambiguous':0,'missing':0,
        'temporal_proposals':0,'encoder_forwards':0,'seconds':0.,'detection_forwards':0,
        'sam_forwards':0,'detection_budget_exhausted_objects':0}
for ep in episodes:
    bound=json.loads((Path(args.anchors)/ep/'association.json').read_text())
    if any(r['status']!='matched' for r in bound['objects'].values()):raise ValueError('unresolved anchor')
    initial_path=Path(args.initial_evidence)/ep/'frame0.png'
    initial=np.asarray(Image.open(initial_path).convert('RGB')).astype(np.int16)
    with np.load(Path(args.anchors)/ep/'masks.npz') as a:anchor={n:a[n] for n in a.files}
    probe=json.loads((Path(args.probes)/ep/'probe.json').read_text())
    names=sorted(bound['objects'])
    keep=bank['episodes']!=ep
    for step in probe['frames']:
        t0=time.perf_counter()
        frame=f'{ep}__f{step:03d}'
        image_path=Path(args.evidence)/frame/'frame0.png'
        image=Image.open(image_path).convert('RGB');rgb=np.asarray(image).astype(np.int16)
        old=Path(args.old_masks)/frame
        masks={};candidates=[]
        if (old/'masks.npz').is_file():
            perception=json.loads((old/'perception.json').read_text())
            with np.load(old/'masks.npz') as a:
                for name in sorted(a.files):
                    mask=a[name]
                    if not mask.any():continue
                    ys,xs=np.nonzero(mask);cid=len(candidates)
                    candidates.append({'cid':cid,'obj':name,'score':float(perception[name]['score']),
                        'box':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],'filtered':None,
                        'source':'frozen_probe_detector'})
                    masks[cid]=mask
        unchanged=(np.max(np.abs(rgb-initial),axis=-1)<=12)
        anchored=set()
        for name,mask in anchor.items():
            retained=float(unchanged[mask].mean()) if mask.any() else 0.
            if retained<.9:continue
            current=mask&unchanged
            ys,xs=np.nonzero(current);cid=len(candidates)
            candidates.append({'cid':cid,'obj':name,'score':float(bound['objects'][name]['score']),
                'box':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],'filtered':None,
                'source':'temporal_rgb_anchor','retained_fraction':retained})
            masks[cid]=current;totals['temporal_proposals']+=1
            anchored.add(name)
        unresolved=sorted(set(names)-anchored)
        totals['detection_budget_exhausted_objects']+=max(0,len(unresolved)-3)
        raw=[]
        for name in unresolved[:3]:
            obj=object_specs[name]
            for prompt in prompts_for(obj,world.get('materials',{}).get(obj.get('material'))):
                inp=proc(images=image.resize((image.width*3,image.height*3)),text=prompt+'.',return_tensors='pt').to('cuda')
                with torch.no_grad():prediction=net(**inp)
                totals['detection_forwards']+=1
                detections=proc.post_process_grounded_object_detection(prediction,inp.input_ids,threshold=.1,
                    text_threshold=.1,target_sizes=[(image.height,image.width)])[0]
                for score,box in zip(detections['scores'].tolist(),detections['boxes'].tolist()):
                    why=filter_reason(box,name,prior)
                    if why is None:raw.append({'obj':name,'score':score,'box':box,'prompt':prompt,'filtered':None})
        if raw:
            inp=sam_proc(images=image,input_boxes=[[r['box'] for r in raw]],return_tensors='pt').to('cuda')
            with torch.no_grad():prediction=sam(**inp,multimask_output=False)
            new_masks=sam_proc.post_process_masks(prediction.pred_masks.cpu(),inp['original_sizes'].cpu())[0]
            totals['sam_forwards']+=1
            for index,candidate in enumerate(raw):
                mask=new_masks[index,0].numpy().astype(bool)
                if not mask.any():continue
                cid=len(candidates)
                candidates.append({**candidate,'cid':cid,'source':'bounded_changed_region_redetection'})
                masks[cid]=mask
        if not candidates:raise ValueError('no candidate masks for frame')
        inputs=proc(images=image.resize((image.width*3,image.height*3)),text='object.',return_tensors='pt').to('cuda')
        with torch.no_grad():
            feature,_=net.model.backbone(inputs['pixel_values'],inputs['pixel_mask'])
            vectors=descriptors(feature,[masks[c['cid']] for c in candidates])
        preds=classify(vectors,bank['features'][keep],bank['labels'][keep].tolist(),names)
        result=reassociate(names,candidates,masks,dict(enumerate(preds)),share_all=True,
                           prior_filter=lambda box,name:filter_reason(box,name,prior))
        dest=out/frame;dest.mkdir()
        np.savez_compressed(dest/'masks.npz',**{n:result['masks'][r['cid']] for n,r in result['objects'].items()
                                             if r['cid'] is not None})
        metadata={'schema':1,'episode':frame,'algorithm':'temporal_visual_probe_v2','objects':result['objects'],
                  'masks_sha256':sha(dest/'masks.npz'),'source':{'image_sha256':sha(image_path),
                    'anchor_sha256':sha(Path(args.anchors)/ep/'association.json'),'bank_sha256':sha(args.bank),
                    'code_sha256':sha(__file__)},
                  'temporal_rule':{'rgb_linf_max':12,'retained_fraction_min':.9}}
        (dest/'association.json').write_text(json.dumps(metadata,indent=1))
        (dest/'identity_audit.json').write_text(json.dumps({'candidates':candidates,'visual_predictions':preds,
               'removed':result['rejected'],'added':result['added']},indent=1))
        totals['frames']+=1;totals['objects']+=len(names);totals['encoder_forwards']+=1
        for r in result['objects'].values():totals[r['status']]+=1
        totals['seconds']+=time.perf_counter()-t0
    print('episodes_complete',totals['frames'],flush=True)
(out/'identity_summary.json').write_text(json.dumps(totals,indent=2))
print(json.dumps(totals))
