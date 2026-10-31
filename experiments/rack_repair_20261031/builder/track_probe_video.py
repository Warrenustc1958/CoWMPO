# Copyright 2026 The RLinf Authors.
"""SAM2 video propagation from verified first-frame identities; Builder RGB only."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import Sam2VideoModel,Sam2VideoProcessor
from cowm import roles

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=argparse.ArgumentParser()
for key in ['anchors','initial-evidence','evidence','probes','out','data-role']:
    parser.add_argument('--'+key,required=True)
parser.add_argument('--episodes',default='')
a=parser.parse_args()
if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise RuntimeError('GPU2 only')
eps=a.episodes.split(',') if a.episodes else sorted(p.name for p in Path(a.anchors).iterdir() if p.is_dir())
if a.data_role not in ['dev','fit']:raise ValueError('dev/fit only')
roles.require(eps,{a.data_role},'SAM2 video identity propagation')
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
model_path='/data_cf/cowmpo/builder_assets/models/sam2.1-hiera-large'
processor=Sam2VideoProcessor.from_pretrained(model_path)
model,loading=Sam2VideoModel.from_pretrained(model_path,output_loading_info=True)
if any(loading.get(key) for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
    raise ValueError('video checkpoint did not load exactly: '+str(loading))
model=model.cuda().eval()
totals={'episodes':len(eps),'frames':0,'objects':0,'matched':0,'ambiguous':0,'missing':0,'seconds':0.}
for ep in eps:
    t0=time.perf_counter();anchor_root=Path(a.anchors)/ep
    bound=json.loads((anchor_root/'association.json').read_text())
    if sha(anchor_root/'masks.npz')!=bound['masks_sha256']:raise ValueError('anchor hash mismatch')
    if any(r['status']!='matched' for r in bound['objects'].values()):raise ValueError('unresolved anchor')
    names=sorted(bound['objects']);probe=json.loads((Path(a.probes)/ep/'probe.json').read_text());steps=probe['frames']
    if not steps or steps[0]!=0:raise ValueError('expected first evidence frame at step zero')
    frames=[Image.open(Path(a.evidence)/f'{ep}__f{step:03d}'/'frame0.png').convert('RGB') for step in steps]
    initial=Image.open(Path(a.initial_evidence)/ep/'frame0.png').convert('RGB')
    with np.load(anchor_root/'masks.npz') as archive:anchors=np.stack([archive[n] for n in names])
    delta=np.max(np.abs(np.asarray(initial).astype(np.int16)-np.asarray(frames[0]).astype(np.int16)),axis=-1)
    alignment={'changed_pixel_fraction_gt25':float((delta>25).mean()),
               'per_object_changed_fraction_gt25':{n:float((delta[anchors[i]]>25).mean()) for i,n in enumerate(names)}}
    if alignment['changed_pixel_fraction_gt25']>.005 or any(v>.2 for v in alignment['per_object_changed_fraction_gt25'].values()):
        raise ValueError('anchor and probe initialization differ beyond alignment tolerance: '+ep+' '+str(alignment))
    session=processor.init_video_session(video=frames,inference_device='cuda',processing_device='cpu',
         video_storage_device='cpu',dtype=torch.float32)
    processor.add_inputs_to_inference_session(session,frame_idx=0,obj_ids=list(range(len(names))),input_masks=list(anchors))
    with torch.no_grad():
        for result in model.propagate_in_video_iterator(session,start_frame_idx=0,show_progress_bar=False):
            index=int(result.frame_idx);step=steps[index];frame=f'{ep}__f{step:03d}';dest=out/frame;dest.mkdir()
            masks=processor.post_process_masks([result.pred_masks],original_sizes=[frames[index].size[::-1]],
                 apply_non_overlapping_constraints=True)[0].cpu().numpy().astype(bool)
            scores=result.object_score_logits.detach().float().cpu().numpy().reshape(-1)
            records={};by_name={}
            for row,obj_id in enumerate(result.object_ids):
                name=names[obj_id];mask=masks[row,0] if masks.ndim==4 else masks[row]
                status='matched' if bool(mask.any()) and scores[row]>0 else 'missing'
                if index==0:mask=anchors[obj_id];status='matched'
                records[name]={'status':status,'cid':int(obj_id) if mask.any() else None,
                    'visibility_logit':float(scores[row]),'identity_source':'verified initial mask + video memory',
                    'reason':None if status=='matched' else 'tracker reports absent or empty'}
                if mask.any():by_name[name]=mask
                totals[status]+=1
            if set(records)!=set(names):raise ValueError('tracker omitted an object identity')
            np.savez_compressed(dest/'masks.npz',**by_name)
            record={'schema':1,'episode':frame,'algorithm':'sam2_video_v1','objects':records,
              'masks_sha256':sha(dest/'masks.npz'),'source':{'anchor_sha256':sha(anchor_root/'association.json'),
               'image_sha256':sha(Path(a.evidence)/frame/'frame0.png'),'code_sha256':sha(__file__),
               'model_config_sha256':sha(Path(model_path)/'config.json')},
              'protocol':{'presence_logit_threshold':0.,'non_overlapping_masks':True,'sparse_frame_steps':steps,
                          'initial_alignment':alignment,'max_global_changed_fraction':.005,
                          'max_object_changed_fraction':.2,'pixel_difference_threshold':25}}
            (dest/'association.json').write_text(json.dumps(record,indent=1));totals['frames']+=1;totals['objects']+=len(names)
    torch.cuda.synchronize();totals['seconds']+=time.perf_counter()-t0
    del session
    print('episodes_complete',totals['frames'],flush=True)
(out/'identity_summary.json').write_text(json.dumps(totals,indent=2));print(json.dumps(totals))
