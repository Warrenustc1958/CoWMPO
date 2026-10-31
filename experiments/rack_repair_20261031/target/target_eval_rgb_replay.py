"""Target-side evaluation of RGB-initialized fixed-action replay."""
import json
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');OUT=R/'grasp_rgb_robot_dev_v1'
rows=[];pairs=[]
for p in sorted(OUT.glob('*/result.json')):
    r=json.loads(p.read_text());key=p.parent.name
    if 'prediction' in r:
        label=json.loads((T/'grasp_contrast_dev_v1'/(key+'.json')).read_text())['label'];pairs.append((label,r['prediction']))
        row={'key':key,'label':label,'prediction':r['prediction']}
    else:row={'key':key}
    gt=T/'contact_diagnosis_dev_v1'/(key+'_eef.npy')
    if gt.exists():
        error=np.linalg.norm(np.load(p.parent/'eef.npy')-np.load(gt),axis=1)*1000
        row.update(eef_mm_step60=float(error[59]),eef_mm_step130=float(error[129]),eef_mm_step240=float(error[239]))
    rows.append(row)
tp=sum(y and p for y,p in pairs);tn=sum(not y and not p for y,p in pairs);pos=sum(y for y,p in pairs);neg=len(pairs)-pos
report={'queries':len(pairs),'TP':tp,'TN':tn,'FP':neg-tn,'FN':pos-tp,'balanced_accuracy':(tp/pos+tn/neg)/2,'rows':rows}
(T/'grasp_rgb_robot_dev_v1.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
