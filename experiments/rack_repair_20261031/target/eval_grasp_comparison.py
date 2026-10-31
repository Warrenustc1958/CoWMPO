# Copyright 2026 The RLinf Authors.
"""Target-only paired comparison on the twelve dev grasp contrasts."""
import json
from pathlib import Path
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/grasp_contrast_dev_v1')
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020/grasp_reference_dev_v1')
rows=[]
for p in sorted(T.glob('libero_goal_*.json')):
    row=json.loads(p.read_text());baseline=json.loads((R/p.name).read_text());row['reference_prediction']=baseline['grasped'];rows.append(row)
if len(rows)!=12:raise ValueError('incomplete paired comparison')
report={}
for name,key in [('shape','prediction'),('reference','reference_prediction')]:
    tp=sum(r['label'] and r[key] for r in rows);tn=sum(not r['label'] and not r[key] for r in rows)
    pos=sum(r['label'] for r in rows);neg=len(rows)-pos
    report[name]={'TP':tp,'TN':tn,'FP':neg-tn,'FN':pos-tp,'balanced_accuracy':(tp/pos+tn/neg)/2 if pos and neg else None}
report['scope']='same twelve actions at six dev initial states; not independent P1 validation'
report['reference_note']='reference template with ceramic.friction=0.815; cream_cheese material unaffected by this override'
(T/'comparison.json').write_text(json.dumps({'summary':report,'rows_dev':rows},indent=1));print(json.dumps(report,indent=1))
