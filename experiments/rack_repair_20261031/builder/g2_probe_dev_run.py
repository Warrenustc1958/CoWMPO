# Copyright 2026 The RLinf Authors.
"""Diagnostic dev G2 with frozen frame identities; shard construction/replay in isolated renderers."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cowm import roles
from cowm.s6.g2_replay import build_validity,evaluate,summarize,verify_manifest,require_reference_bindings
from cowm.world.artifact import tree_sha

R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'g2_probe_dev_v4'
WORLD=Path('/data_cf/cowmpo/worlds/s5_v4/shape/world_fit')
STATE=R/'state_visual_dev_merged_v1/dev3'
PROBES=Path('/data_cf/cowmpo/worlds/s6_strike/dev3')
MASKS=R/'probe_identity_dev_v4'
PAIR=R/'world_pair_v1'
EPS=sorted(p.name for p in STATE.iterdir() if p.is_dir())
roles.require(EPS,{'dev'},'diagnostic G2 replay')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase');parser.add_argument('--part',type=int)
    parser.add_argument('--variant');a=parser.parse_args()
    if a.phase=='build':
        build_validity(str(WORLD),str(STATE),str(PROBES),str(MASKS),EPS[a.part::6],str(OUT/f'manifest_part{a.part}.json'))
        return
    variants={'legacy':(WORLD,STATE,{}),'pair_prior':(PAIR,OUT/'pair_states',{}),
              'pair_low':(PAIR,OUT/'pair_states',{'ceramic.friction':.4}),
              'pair_high':(PAIR,OUT/'pair_states',{'ceramic.friction':1.1})}
    if a.phase=='evaluate':
        world,state,params=variants[a.variant]
        result=evaluate(str(world),str(state),str(PROBES),str(MASKS),EPS[a.part::6],params=params,
                        validity_path=str(OUT/'validity.json'),world_tree_sha256=tree_sha(world),require_bound_reference=True)
        (OUT/f'{a.variant}_part{a.part}.json').write_text(json.dumps(result,indent=1))
        return
    if not (MASKS/'identity_summary.json').is_file():raise ValueError('probe identity run incomplete')
    OUT.mkdir(exist_ok=False)
    pair_meta=json.loads((PAIR/'world.json').read_text());pair_hash=tree_sha(PAIR)
    for ep in EPS:
        path=STATE/ep/'state.json';state=json.loads(path.read_text())
        state.update(world_sha256=pair_meta['world_sha256'],world_tree_sha256=pair_hash,
                     derivation={'change':'contact mixing only; geometry and estimated poses unchanged',
                                 'source_state_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        target=OUT/'pair_states'/ep;target.mkdir(parents=True)
        (target/'state.json').write_text(json.dumps(state,indent=1))
    def run(job):
        phase,part,variant=job;name=f'{phase}_{variant or ""}_{part}'
        cmd=[sys.executable,__file__,'--phase',phase,'--part',str(part)]
        if variant:cmd+=['--variant',variant]
        with (OUT/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((name,result.returncode))
        print('complete',name,flush=True)
    with ThreadPoolExecutor(6) as pool:list(pool.map(run,[('build',i,None) for i in range(6)]))
    parts=[json.loads((OUT/f'manifest_part{i}.json').read_text()) for i in range(6)]
    manifest={k:v for k,v in parts[0].items() if k!='episodes'};manifest['episodes']={}
    for part in parts:
        for ep,record in part['episodes'].items():
            if ep in manifest['episodes']:raise ValueError('duplicate manifest episode')
            manifest['episodes'][ep]=record
    if set(manifest['episodes'])!=set(EPS):raise ValueError('manifest episode set mismatch')
    require_reference_bindings(manifest,EPS);verify_manifest(manifest,str(PROBES),str(MASKS),EPS)
    (OUT/'validity.json').write_text(json.dumps(manifest,indent=1))
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(run,[('evaluate',i,name) for name in variants for i in range(6)]))
    report={}
    for name in variants:
        results=[r for i in range(6) for r in json.loads((OUT/f'{name}_part{i}.json').read_text())]
        if len(results)!=len(EPS) or {r['episode'] for r in results}!=set(EPS):raise ValueError('replay set mismatch')
        report[name]=summarize(results)
    (OUT/'report.json').write_text(json.dumps({'role':'dev diagnostic only; no parameter selection',
         'episodes':EPS,'variants':report},indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
