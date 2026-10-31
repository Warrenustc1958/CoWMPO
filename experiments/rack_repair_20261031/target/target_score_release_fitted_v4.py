# Copyright 2026 The RLinf Authors.
"""Private all-variant release comparison; keep acquisition failures and event timing."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');LABELS=T/'release_event_dev_v3';NEW=R/'release_fitted_dev_v4';OLD=R/'release_event_dev_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((NEW/'freeze.json').read_text());keys=cfg['keys']
if json.loads((NEW/'complete.json').read_text())['keys']!=keys:raise ValueError('incomplete fitted replay')
if sha(R/'s6_natural_v3/fit/s6_report.json')!=cfg['s6_report_sha256']:raise ValueError('fitted parameter source changed')
report={'queries':len(keys),'scope':'six exposed dev episodes, paired low/high actions; same frozen S6 fits, no further fitting; event labels corrected after dev diagnosis; not independent P1 validation','models':{},'sources':{}}
for name,source in [('reference',OLD),('shape',OLD),('param',NEW),('param_struct',NEW)]:
    rows=[];time_errors=[];mismatches=[]
    for key in keys:
        label_path=LABELS/(key+'.json');prediction_path=source/key/(name+'.json');label=json.loads(label_path.read_text());prediction=json.loads(prediction_path.read_text())
        if label['query_sha256']!=prediction['query_sha256']:raise ValueError('query mismatch')
        y=label['label'];p=prediction['prediction'];rows.append((y['status'],p['status']))
        if y['status']!=p['status']:mismatches.append(key)
        if y['acquired'] and p['acquired'] and y.get('event_step') is not None and p.get('event_step') is not None:time_errors.append({'key':key,'target_step':y['event_step'],'prediction_step':p['event_step'],'absolute_steps':abs(y['event_step']-p['event_step'])})
        report['sources'][str(label_path)]=sha(label_path);report['sources'][str(prediction_path)]=sha(prediction_path)
    counts=Counter(rows);recall={label:sum(y==label and p==label for y,p in rows)/sum(y==label for y,p in rows) for label in ['stable','dropped'] if any(y==label for y,p in rows)}
    report['models'][name]={'confusion':[{'label':y,'prediction':p,'n':n} for (y,p),n in sorted(counts.items())],'all_query_accuracy':sum(y==p for y,p in rows)/len(rows),
        'conditional_release_ba':float(np.mean(list(recall.values()))) if len(recall)==2 else None,'conditional_release_denominator':sum(y in ['stable','dropped'] for y,p in rows),
        'acquisition_failures_retained':sum(y=='no_carry' for y,p in rows),'wrong_keys':mismatches,
        'onset_error_steps_mean':float(np.mean([r['absolute_steps'] for r in time_errors])) if time_errors else None,'onset_error_steps_max':max([r['absolute_steps'] for r in time_errors],default=None),'onset_comparisons':time_errors}
(T/'release_fitted_dev_v4_score.json').write_text(json.dumps(report,indent=1));print(json.dumps({'queries':report['queries'],'models':{k:{n:v for n,v in d.items() if n!='onset_comparisons'} for k,d in report['models'].items()}},indent=1))
