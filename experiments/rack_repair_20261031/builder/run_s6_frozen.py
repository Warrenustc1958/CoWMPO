# Copyright 2026 The RLinf Authors.
"""Frozen SAM2-video evidence + fit-only S6 CMA-ES; gate/test never read."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cowm import roles
from cowm.s6.g2_replay import build_validity,verify_manifest,require_reference_bindings
from cowm.world.artifact import tree_sha

SC=Path(__file__).parent
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
READY=R/'s6_ready_v1'
OUT=R/'s6_identification_v1'
SHAPE=R/'world_pair_v1'
REF=READY/'reference_world_pair'
PROBES=READY/'probes'
EPS=sorted(p.name for p in PROBES.iterdir() if p.is_dir())
roles.require(EPS,{'dev','fit'},'S6 frozen experiment')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build-part',type=int);a=parser.parse_args()
    if a.build_part is not None:
        build_validity(str(SHAPE),str(READY/'shape_states'),str(PROBES),str(OUT/'probe_masks'),
                       EPS[a.build_part::12],str(OUT/f'validity_part{a.build_part}.json'))
        return
    if len(roles.select(EPS,'fit'))!=28 or len(roles.select(EPS,'dev'))!=32:
        raise ValueError('expected exact 28 fit / 32 dev roles')
    frozen=json.loads((R/'video_freeze_v1.json').read_text())
    for p,digest in frozen['files'].items():
        if sha(p)!=digest:raise ValueError('frozen perception code/model changed: '+p)
    sources=[R/'probe_video_dev_v1',R/'probe_video_fit_v1',R/'probe_video_exposed_v1']
    expected={f'{ep}__f{s:03d}' for ep in EPS for s in json.loads((PROBES/ep/'probe.json').read_text())['frames']}
    for source in sources:
        if not (source/'identity_summary.json').is_file():raise ValueError('incomplete video inference: '+str(source))
    OUT.mkdir(exist_ok=False);masks=OUT/'probe_masks';masks.mkdir()
    for source in sources:
        for p in sorted(source.glob('*/association.json')):shutil.copytree(p.parent,masks/p.parent.name)
    if {p.name for p in masks.iterdir()}!=expected:raise ValueError('probe frame set mismatch')
    reference_states=OUT/'reference_states';reference_states.mkdir()
    for role in ['dev','fit']:
        for p in (READY/('reference_states_'+role)).glob('libero_goal_*/state.json'):
            shutil.copytree(p.parent,reference_states/p.parent.name)
    if {p.name for p in reference_states.iterdir()}!=set(EPS):raise ValueError('reference-state set mismatch')
    config={'roles':{'fit':28,'dev':32},'fit_episodes':roles.select(EPS,'fit'),'dev_episodes':roles.select(EPS,'dev'),
       'reference_hash':tree_sha(REF),'shape_hash':tree_sha(SHAPE),'perception_freeze_sha256':sha(R/'video_freeze_v1.json'),
       'fitter_sha256':sha(SC/'toolkits/cowm/s6/s6_identify_v2.py'),
       'replay_sha256':sha(SC/'toolkits/cowm/s6/g2_replay.py'),
       'optimizer':{'name':'CMA-ES','generations':6,'popsize':6,'seed':0,'parameters':['ceramic.friction']},
       'identifiability_sha256':sha(R/'ident_dev_v1/pair.json'),'gate_test_queried':False}
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2))
    def build(i):
        with (OUT/f'build_part{i}.log').open('w') as log:
            p=subprocess.run([sys.executable,__file__,'--build-part',str(i)],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(('manifest',i,p.returncode))
        print('manifest_part_complete',i,flush=True)
    with ThreadPoolExecutor(12) as pool:list(pool.map(build,range(12)))
    parts=[json.loads((OUT/f'validity_part{i}.json').read_text()) for i in range(12)]
    manifest={k:v for k,v in parts[0].items() if k!='episodes'};manifest['episodes']={}
    for part in parts:
        if set(manifest['episodes'])&set(part['episodes']):raise ValueError('duplicate manifest episodes')
        manifest['episodes'].update(part['episodes'])
    if set(manifest['episodes'])!=set(EPS):raise ValueError('manifest episode set mismatch')
    require_reference_bindings(manifest,EPS);verify_manifest(manifest,str(PROBES),str(masks),EPS)
    validity=OUT/'validity.json';validity.write_text(json.dumps(manifest,indent=1))
    cmd=[sys.executable,str(SC/'toolkits/cowm/s6/s6_identify_v2.py'),
       '--ref-world',str(REF),'--ref-state',str(reference_states),
       '--shape-world',str(SHAPE),'--shape-state',str(READY/'shape_states'),
       '--ref-tree-sha256',config['reference_hash'],'--shape-tree-sha256',config['shape_hash'],
       '--probes',str(PROBES),'--probe-masks',str(masks),'--validity',str(validity),
       '--ident',str(R/'ident_dev_v1/pair.json'),'--out',str(OUT/'fit'),
       '--generations','6','--popsize','6','--jobs','24']
    completed=subprocess.run(cmd)
    if completed.returncode:raise RuntimeError('S6 fitting failed: '+str(completed.returncode))
    if tree_sha(REF)!=config['reference_hash'] or tree_sha(SHAPE)!=config['shape_hash']:
        raise ValueError('frozen world changed during fitting')
    print('S6_FROZEN_COMPLETE',flush=True)

if __name__=='__main__':main()
