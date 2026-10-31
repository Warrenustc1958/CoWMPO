# Copyright 2026 The RLinf Authors.
"""Revalidate S6 with frozen supported geometry and RGB-estimated robot states."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cowm import roles
from cowm.world.artifact import tree_sha,verify_tree
from cowm.s6.g2_replay import build_validity,verify_manifest,require_reference_bindings
SC=Path(__file__).parent
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'s6_natural_v2';READY=R/'s6_ready_v1';SHAPE=R/'bowl_support_dev_v1/world11';REF=READY/'reference_world_pair';PROBES=READY/'probes';MASKS=R/'s6_identification_v1/probe_masks'
EPS=sorted(p.name for p in PROBES.iterdir() if p.is_dir())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--build-part',type=int);p.add_argument('--ident-param');a=p.parse_args()
    if a.build_part is not None:
        build_validity(str(SHAPE),str(OUT/'shape_states'),str(PROBES),str(MASKS),EPS[a.build_part::8],str(OUT/f'validity_part{a.build_part}.json'));return
    if a.ident_param:
        cmd=[sys.executable,str(SC/'toolkits/cowm/s6/identifiability.py'),'--world',str(SHAPE),'--state',str(OUT/'shape_states'),'--probes',str(PROBES),'--out',str(OUT/(a.ident_param+'.json')),'--params',a.ident_param,'--episodes',','.join(f'libero_goal_t{k}_i0' for k in range(6))]
        subprocess.run(cmd,check=True);return
    if len(roles.select(EPS,'fit'))!=28 or len(roles.select(EPS,'dev'))!=32:raise ValueError('exact roles required')
    OUT.mkdir(exist_ok=False);frozen=json.loads((R/'video_freeze_v1.json').read_text())
    for path,h in frozen['files'].items():
        if sha(path)!=h:raise ValueError('perception code/model changed')
    source_hashes={}
    for ep in EPS:
        role=roles.role(ep);roles.require([ep],{'dev','fit'},'RGB S6 initialization')
        rob=R/('robot_rgb_fit_v2' if role=='fit' else 'robot_rgb_all_dev_v2')/ep/'robot.json'
        for label,world,source in [('shape',SHAPE,R/('natural_bowl_fit_v2' if role=='fit' else 'natural_states_all_dev_v1')/ep/'state.json'),('reference',REF,READY/('reference_states_'+role)/ep/'state.json')]:
            doc=json.loads(source.read_text());doc.update(robot=json.loads(rob.read_text())['robot'],robot_required=True,world_sha256=json.loads((world/'world.json').read_text())['world_sha256'],world_tree_sha256=tree_sha(world))
            doc['initialization_provenance']={'object_source_sha256':sha(source),'robot_record_sha256':sha(rob),'robot_source':'initial RGB estimate; zero-velocity prior; shared reset integrator'}
            dest=OUT/(label+'_states')/ep;dest.mkdir(parents=True);(dest/'state.json').write_text(json.dumps(doc,indent=1));source_hashes.update({str(q):sha(q) for q in [source,rob]})
    config={'roles':{'fit':28,'dev':32},'shape_hash':tree_sha(SHAPE),'reference_hash':tree_sha(REF),'sources':source_hashes,'code':{str(SC/n):sha(SC/n) for n in ['toolkits/cowm/init/robot_state.py','toolkits/cowm/init/estimate_state.py','toolkits/cowm/s6/g2_replay.py','toolkits/cowm/s6/identifiability.py','toolkits/cowm/s6/s6_identify_v2.py',Path(__file__).name]},'perception_freeze_sha256':sha(R/'video_freeze_v1.json'),'GT_inputs':False,'gate_test_queried':False,'qualification':'adaptive dev; both template and structural models use same RGB robot estimates; same frozen strike actions; not P1 grasp/release certification'}
    (OUT/'config.json').write_text(json.dumps(config,indent=1))
    def run(flag,value):
        with (OUT/(str(value)+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,flag,str(value)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((flag,value,r.returncode))
        print('S6_NATURAL_PREP_COMPLETE',flag,value,flush=True)
    with ThreadPoolExecutor(8) as pool:list(pool.map(lambda i:run('--build-part',i),range(8)))
    parts=[json.loads((OUT/f'validity_part{i}.json').read_text()) for i in range(8)];manifest={k:v for k,v in parts[0].items() if k!='episodes'};manifest['episodes']={}
    for part in parts:
        if set(manifest['episodes'])&set(part['episodes']):raise ValueError('duplicate episodes')
        manifest['episodes'].update(part['episodes'])
    if set(manifest['episodes'])!=set(EPS):raise ValueError('manifest episode mismatch')
    require_reference_bindings(manifest,EPS);verify_manifest(manifest,str(PROBES),str(MASKS),EPS);(OUT/'validity.json').write_text(json.dumps(manifest,indent=1))
    params=['tabletop.friction','ceramic.density','ceramic.friction']
    with ThreadPoolExecutor(3) as pool:list(pool.map(lambda n:run('--ident-param',n),params))
    ident={}
    for n in params:ident.update(json.loads((OUT/(n+'.json')).read_text()))
    (OUT/'ident.json').write_text(json.dumps(ident,indent=1));print('S6_NATURAL_IDENT',json.dumps(ident),flush=True)
    from cowm.s6.s6_identify_v2 import g2_eval
    with ThreadPoolExecutor(8) as pool:
        prior={}
        for role in ['fit','dev']:
            summary,episodes=g2_eval(pool,str(SHAPE),str(OUT/'shape_states'),str(PROBES),str(MASKS),roles.select(EPS,role),{},str(OUT),config['shape_hash'],str(OUT/'validity.json'),chunks=8)
            prior[role]={'summary':summary,'episodes':episodes};print('S6_NATURAL_STRUCT_PRIOR',role,json.dumps(summary),flush=True)
    (OUT/'struct_prior.json').write_text(json.dumps(prior,indent=1))
    args=[sys.executable,str(SC/'toolkits/cowm/s6/s6_identify_v2.py')]
    for k,v in {'ref-world':REF,'ref-state':OUT/'reference_states','shape-world':SHAPE,'shape-state':OUT/'shape_states','ref-tree-sha256':config['reference_hash'],'shape-tree-sha256':config['shape_hash'],'probes':PROBES,'probe-masks':MASKS,'validity':OUT/'validity.json','ident':OUT/'ident.json','out':OUT/'fit','generations':6,'popsize':6,'jobs':24}.items():args+=['--'+k,str(v)]
    subprocess.run(args,check=True)
    for path,h in {**config['sources'],**config['code']}.items():
        if sha(path)!=h:raise ValueError('frozen input/code changed')
    verify_tree(SHAPE,config['shape_hash']);verify_tree(REF,config['reference_hash']);print('S6_NATURAL_V2_COMPLETE',flush=True)
if __name__=='__main__':main()
