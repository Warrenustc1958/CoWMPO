"""Leave-one-episode-out visual identity diagnostics with v3 pseudo-labels, never GT."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

VISUAL_MARGIN = 0.05
NEIGHBORS = 3


def normalize(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def classify(features, bank_features, bank_labels, names):
    prototypes = np.stack([normalize(bank_features[np.array(bank_labels)==n].mean(0)) for n in names])
    scores = features @ prototypes.T
    order = np.argsort(-scores,axis=1,kind='stable')
    labels = np.array(bank_labels)
    neighbors = labels[np.argsort(-(features @ bank_features.T),axis=1,kind='stable')[:,:NEIGHBORS]]
    records = []
    for i,ranking in enumerate(order):
        name,runner = names[ranking[0]],names[ranking[1]]
        gap = float(scores[i,ranking[0]]-scores[i,ranking[1]])
        agree = bool(np.all(neighbors[i]==name))
        records.append({'label':name,'runner':runner,'gap':gap,'score':float(scores[i,ranking[0]]),
                        'scores':{n:float(scores[i,j]) for j,n in enumerate(names)},
                        'neighbors':neighbors[i].tolist(),'strong':gap>=VISUAL_MARGIN and agree})
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--features',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    root=Path(args.features)
    episodes=[]
    bank_f,bank_n,bank_ep=[],[],[]
    for path in sorted(root.glob('*/*.json')):
        rec=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as data:
            features,cids=data['features'],data['cids'].tolist()
        item={'meta':rec,'features':features,'cids':cids}
        episodes.append(item)
        for n,r in rec['base'].items():
            if r['status']=='matched':
                bank_f.append(features[cids.index(r['cid'])]);bank_n.append(n);bank_ep.append(rec['episode'])
    bank_f=np.stack(bank_f);bank_ep=np.array(bank_ep);names=sorted(set(bank_n))
    counts={n:Counter() for n in names}
    uncertain=[]
    for item in episodes:
        meta,cids=item['meta'],item['cids']
        keep=bank_ep!=meta['episode']
        predictions=classify(item['features'],bank_f[keep],np.array(bank_n)[keep].tolist(),names)
        for n,r in meta['base'].items():
            if r['cid'] is None:
                continue
            prediction=predictions[cids.index(r['cid'])]
            if r['status']=='matched':
                counts[n]['matched']+=1
                counts[n]['top1_same' if prediction['label']==n else 'top1_other']+=1
                counts[n]['strong_same' if prediction['label']==n and prediction['strong'] else 'not_strong_same']+=1
            else:
                alt=predictions[cids.index(r['alt_cid'])] if r['alt_cid'] is not None else None
                uncertain.append({'episode':meta['episode'],'obj':n,'chosen':prediction,'alt':alt})
        item['predictions']=predictions
        (Path(args.out).parent/(meta['episode']+'_visual.json')).write_text(json.dumps(
            {'episode':meta['episode'],'cids':cids,'predictions':predictions},indent=1))
    report={'protocol':{'margin':VISUAL_MARGIN,'neighbors':NEIGHBORS,'bank_labels':'v3 matched only',
                        'validation':'leave-one-episode-out','GT_used':False},
            'trusted_LOO':{n:dict(c) for n,c in counts.items()},'uncertain':uncertain}
    Path(args.out).write_text(json.dumps(report,indent=2))
    np.savez_compressed(Path(args.out).with_suffix('.npz'),features=bank_f,labels=np.array(bank_n),episodes=bank_ep)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
