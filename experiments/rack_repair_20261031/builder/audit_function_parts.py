# Copyright 2026 The RLinf Authors.
"""Dev-only functional-part proposal audit from RGB and declared stove joint."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model
from cowm import roles

R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'functional_parts_dev_v1'
EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
roles.require(EPS,{'dev'},'functional-part RGB audit')
if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise RuntimeError('GPU2 only')
OUT.mkdir(exist_ok=False)
M=Path('/data_cf/cowmpo/builder_assets/models')
gp=AutoProcessor.from_pretrained(M/'grounding-dino-base')
gm=AutoModelForZeroShotObjectDetection.from_pretrained(M/'grounding-dino-base').cuda().eval()
sp=Sam2Processor.from_pretrained(M/'sam2.1-hiera-large')
sm=Sam2Model.from_pretrained(M/'sam2.1-hiera-large').cuda().eval()
for ep in EPS:
    image=Image.open(Path('/data_cf/cowmpo/evidence/libero_goal_dev3')/ep/'frame0.png').convert('RGB')
    with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:base=a['stove']
    yy,xx=np.nonzero(base);x0,y0,x1,y1=xx.min(),yy.min(),xx.max()+1,yy.max()+1
    # Bound search by the observed object, without target geometry or GT labels.
    margin=max(x1-x0,y1-y0)//2
    roi=(max(0,x0-margin),max(0,y0-margin),min(256,x1+margin),min(256,y1+margin))
    records=[];masks={}
    for prompt in ['stove','stove knob','stove handle']:
        inputs=gp(images=image.resize((768,768)),text=prompt+'.',return_tensors='pt').to('cuda')
        with torch.no_grad():out=gm(**inputs)
        result=gp.post_process_grounded_object_detection(out,inputs.input_ids,threshold=.10,text_threshold=.1,target_sizes=[(256,256)])[0]
        for box,score in zip(result['boxes'].cpu().tolist(),result['scores'].cpu().tolist()):
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            if not (roi[0]<=cx<roi[2] and roi[1]<=cy<roi[3]):continue
            inputs=sp(images=image,input_boxes=[[box]],return_tensors='pt').to('cuda')
            with torch.no_grad():out=sm(**inputs,multimask_output=False)
            mk=sp.post_process_masks(out.pred_masks.cpu(),inputs['original_sizes'].cpu())[0][0,0].numpy().astype(bool)
            cid=len(records);masks['c'+str(cid)]=mk
            records.append({'cid':cid,'prompt':prompt,'box':box,'score':score,'area':int(mk.sum()),
                            'additional_pixels':int((mk&~base).sum()),'base_covered':float((mk&base).sum()/base.sum())})
    dest=OUT/ep;dest.mkdir();np.savez_compressed(dest/'candidate_masks.npz',**masks)
    (dest/'audit.json').write_text(json.dumps({'episode':ep,'roi':list(map(int,roi)),'candidates':records},indent=1))
    canvas=image.copy();draw=ImageDraw.Draw(canvas)
    for rec in records:
        draw.rectangle(rec['box'],outline='red');draw.text(rec['box'][:2],str(rec['cid']),fill='yellow')
    canvas.resize((768,768)).save(dest/'overlay.png')
    print(ep,json.dumps(records),flush=True)
