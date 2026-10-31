# Copyright 2026 The RLinf Authors.
"""Audit exact raw replay sets, frozen worlds and paired S6 fitting effects."""
import hashlib
import json
from pathlib import Path
import numpy as np
from cowm import roles
from cowm.world.artifact import verify_tree
from cowm.s6.g2_replay import summarize
from cowm.s6.s6_identify_v2 import require_episode_set
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'s6_identification_v1';fit=OUT/'fit'
report=json.loads((fit/'s6_report.json').read_text());config=json.loads((OUT/'experiment_config.json').read_text())
freeze=json.loads((R/'video_freeze_v1.json').read_text())
for p,digest in freeze['files'].items():
    if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest:raise ValueError('frozen perception changed: '+p)
for name,key in [('reference_world_pair','reference_hash')]:verify_tree(R/'s6_ready_v1'/name,config[key])
verify_tree(R/'world_pair_v1',config['shape_hash'])
for variant in ['default','param','param+struct']:
    raw=json.loads((fit/(variant.replace('+','_')+'_episodes.json')).read_text())
    for role,summary_key,ep_key in [('fit','fit','fit_episodes'),('dev','heldout','heldout_episodes')]:
        require_episode_set(raw[role],report[ep_key]);roles.require(report[ep_key],{role},'S6 result audit')
        summary=summarize(raw[role])
        if summary!=report['variants'][variant][summary_key]:raise ValueError('raw/summary mismatch: '+variant+'/'+role)
prior=json.loads((R/'s6_shape_prior_v1/report.json').read_text())
fitted=json.loads((fit/'param_struct_episodes.json').read_text());effects={}
rng=np.random.default_rng(20261030)
for role in ['fit','dev']:
    before={r['episode']:r for r in prior[role]['episodes']};after={r['episode']:r for r in fitted[role]}
    if set(before)!=set(after):raise ValueError('paired episodes mismatch')
    episodes=sorted(before);delta=np.array([before[e]['dyn_err_px']-after[e]['dyn_err_px'] for e in episodes])
    cluster={}
    for e,d in zip(episodes,delta):cluster.setdefault(roles.parse(e)[0],[]).append(float(d))
    means=np.array([np.mean(v) for v in cluster.values()])
    boots=means[rng.integers(len(means),size=(10000,len(means)))].mean(axis=1)
    effects[role]={'episodes':len(episodes),'prior_dynamic_px':prior[role]['summary']['dyn_err_px'],
       'fitted_dynamic_px':summarize(fitted[role])['dyn_err_px'],
       'paired_mean_improvement_px':float(delta.mean()),'episodes_improved':int((delta>0).sum()),
       'episodes_worse':int((delta<0).sum()),'max_worsening_px':float(max(0,-delta.min())),
       'task_cluster_mean_improvement_px':float(means.mean()),
       'task_cluster_bootstrap_95ci_px':np.percentile(boots,[2.5,97.5]).tolist(),
       'ci_scope':'descriptive ten-task cluster bootstrap; adaptive known dev, not independent test'}
audit={'raw_summaries_equal':True,'frozen_worlds_unchanged':True,'frozen_perception_unchanged':True,
 'roles':{'fit':len(report['fit_episodes']),'dev':len(report['heldout_episodes'])},
 'params':fitted['params'],'effects':effects,
 'limitations':['effective friction near lower bound; not true Target material recovery',
   'G2 concerns bowl strike replay; does not certify all-object S5, P1 or policy learning',
   'S6 uses world_pair_v1, not the subsequently developed stove geometry']}
(OUT/'audit.json').write_text(json.dumps(audit,indent=1));print(json.dumps(audit,indent=1))
