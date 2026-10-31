"""Compare frozen RGB-only development configurations with private grasp labels."""
import json
from pathlib import Path
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
reports={}
for tag in ['original_edges','support_edges','support_basic']:
    root=R/('grasp_rgb_'+tag+'_dev_v1');frozen=json.loads((root/'freeze.json').read_text());pairs=[]
    for key in frozen['keys']:
        label=json.loads((T/'grasp_contrast_dev_v1'/(key+'.json')).read_text())['label'];pred=json.loads((root/(key+'.json')).read_text())['prediction'];pairs.append((key,label,pred))
    pos=sum(y for k,y,p in pairs);neg=len(pairs)-pos;tp=sum(y and p for k,y,p in pairs);tn=sum(not y and not p for k,y,p in pairs)
    reports[tag]={'queries':len(pairs),'TP':tp,'TN':tn,'FP':neg-tn,'FN':pos-tp,'balanced_accuracy':(tp/pos+tn/neg)/2,'wrong':[k for k,y,p in pairs if y!=p]}
(T/'grasp_rgb_combinations_dev_v1.json').write_text(json.dumps(reports,indent=1));print(json.dumps(reports,indent=1))
