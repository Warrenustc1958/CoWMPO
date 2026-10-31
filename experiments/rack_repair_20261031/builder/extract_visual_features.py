"""Frozen image encoder descriptors for every proposal; Builder-visible RGB/masks only."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from cowm import roles
from cowm.init.associate import associate


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def descriptors(features, masks):
    """Area-weighted mask pooling, equal unit norm per feature scale, final unit norm."""
    masks = torch.as_tensor(np.stack(masks), device=features[0][0].device).float()[:, None]
    parts = []
    for feature, _ in features:
        weights = F.interpolate(masks, size=feature.shape[-2:], mode='area')[:, 0].flatten(1)
        avg = weights @ feature[0].flatten(1).T / weights.sum(1, keepdim=True).clamp_min(1e-8)
        parts.append(F.normalize(avg, dim=-1))
    return F.normalize(torch.cat(parts, dim=-1), dim=-1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    for name in ('pool', 'evidence', 'out', 'world', 'episodes'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--data-role', default='dev', choices=['dev','fit'])
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise RuntimeError('GPU2 only')
    eps = args.episodes.split(',')
    roles.require(eps, {args.data_role}, 'frozen visual feature extraction')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    model_path = '/data_cf/cowmpo/builder_assets/models/grounding-dino-base'
    proc = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).cuda().eval()
    names = [o['name'] for o in json.loads((Path(args.world)/'world.json').read_text())['objects']]
    for ep in eps:
        torch.cuda.synchronize()
        started = time.perf_counter()
        source = Path(args.pool)/ep
        audit = json.loads((source/'audit.json').read_text())
        candidates = audit['candidates']
        with np.load(source/'candidate_masks.npz') as archive:
            masks = {c['cid']: archive[f"c{c['cid']}"] for c in candidates}
        image_path = Path(args.evidence)/ep/'frame0.png'
        image = Image.open(image_path).convert('RGB')
        inp = proc(images=image.resize((image.width*3,image.height*3)),text='object.',return_tensors='pt').to('cuda')
        with torch.no_grad():
            features,_ = model.model.backbone(inp['pixel_values'],inp['pixel_mask'])
            vector = descriptors(features,[masks[c['cid']] for c in candidates])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        np.savez_compressed(output/(ep+'.npz'),features=vector,cids=np.array([c['cid'] for c in candidates]))
        record = {'episode':ep,'base':associate(names,candidates,masks)['objects'],
                  'candidate_sha256':file_sha(source/'candidate_masks.npz'),
                  'audit_sha256':file_sha(source/'audit.json'),'image_sha256':file_sha(image_path),
                  'encoder_config_sha256':file_sha(Path(model_path)/'config.json'),
                  'extractor_sha256':file_sha(__file__),'feature_shape':list(vector.shape),
                  'channels':[int(f[0].shape[1]) for f in features],
                  'seconds':elapsed, 'encoder_forward_passes':1,
                  'timing_scope':'episode IO, encoder, pooling; excludes model initialization'}
        (output/(ep+'.json')).write_text(json.dumps(record))
        print(ep,'features',list(vector.shape),flush=True)


if __name__=='__main__':
    main()
