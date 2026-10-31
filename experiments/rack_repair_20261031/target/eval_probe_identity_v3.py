"""Target-only dev identity/quality evaluation with all frame-object instances retained."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
P=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/probe_identity_dev_v3')
counts=defaultdict(Counter);problems=[]
def iou(a,b):
    u=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum()/u) if u else 0.
for path in sorted(P.glob('*/association.json')):
    record=json.loads(path.read_text());frame=record['episode']
    with np.load(path.parent/'masks.npz') as a:pred={n:a[n] for n in a.files}
    with np.load(T/'probe_gt_dev_v1'/frame/'gt_masks.npz') as a:gt={n:a[n] for n in a.files}
    for name,status in record['objects'].items():
        c=counts[name];c['total']+=1;c[status['status']]+=1
        visible=bool(gt[name].any());c['gt_visible']+=visible
        if not visible:
            c['gt_empty_'+status['status']]+=1
            continue
        if status['status']!='matched':
            c['visible_'+status['status']]+=1
            problems.append({'frame':frame,'name':name,'status':status['status']})
            continue
        overlaps={n:iou(pred[name],mk) for n,mk in gt.items()}
        best=max(overlaps,key=overlaps.get)
        c['matched_mask_iou_ge_05']+=overlaps[name]>=.5
        if best==name and overlaps[name]>=.3:
            c['matched_identity_correct']+=1
        elif best!=name and overlaps[best]>=.3:
            c['matched_identity_wrong']+=1
            problems.append({'frame':frame,'name':name,'status':'wrong','best_gt':best,
                             'named_iou':overlaps[name],'best_iou':overlaps[best]})
        else:
            c['matched_low_overlap']+=1
            problems.append({'frame':frame,'name':name,'status':'low_overlap','best_gt':best,
                             'named_iou':overlaps[name],'best_iou':overlaps[best]})
total=Counter()
for c in counts.values():total.update(c)
report={'definitions':{'identity':'named GT is best IoU match, IoU>=0.3',
                       'mask_quality':'named GT IoU>=0.5','visibility':'at least one GT pixel',
                       'scope':'dev only; GT never input to Builder'},
        'total':dict(total),'per_object':{n:dict(c) for n,c in counts.items()},'problems_dev':problems}
(T/'probe_identity_dev_v3.json').write_text(json.dumps(report,indent=1))
print(json.dumps({k:v for k,v in report.items() if k!='problems_dev'},indent=2))
