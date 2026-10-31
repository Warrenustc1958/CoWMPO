# Copyright 2026 The RLinf Authors.
"""Score an exact frozen grasp query set against Target-private physical labels."""
import argparse
import hashlib
import json
from pathlib import Path
from cowm import roles
p=argparse.ArgumentParser()
for n in ['pred-root','label-root','out']:p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--role',choices=['dev','fit'],required=True);a=p.parse_args()
frozen=json.loads((a.pred_root/'freeze.json').read_text());keys=frozen['keys'];rows=[];sources={}
for k in keys:
    pred_path=a.pred_root/(k+'.json');label_path=a.label_root/(k+'.json');pred=json.loads(pred_path.read_text());label=json.loads(label_path.read_text())
    if pred['episode']!=label['episode'] or pred['key']!=k or label['key']!=k:raise ValueError('query identity mismatch')
    roles.require([pred['episode']],{a.role},'Target grasp evaluation');rows.append((k,label['label'],pred['prediction']));sources.update({str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [pred_path,label_path]})
pos=sum(y for k,y,p in rows);neg=len(rows)-pos;tp=sum(y and p for k,y,p in rows);tn=sum(not y and not p for k,y,p in rows)
report={'queries':len(rows),'positive':pos,'negative':neg,'TP':tp,'TN':tn,'FP':neg-tn,'FN':pos-tp,'balanced_accuracy':(tp/pos+tn/neg)/2 if pos and neg else None,'role':a.role,'scope':'same fixed actions; Target physical labels are private diagnostic only','sources':sources}
if a.role=='dev':report['wrong']=[k for k,y,p in rows if y!=p]
a.out.write_text(json.dumps(report,indent=1));print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=1))
