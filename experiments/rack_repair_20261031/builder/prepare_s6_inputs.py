"""Stage exact 28 fit / 32 dev inputs and frozen reference geometry, without Target internals."""
import hashlib
import json
import shutil
from pathlib import Path
from cowm import roles
from cowm.world.artifact import tree_sha
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
OUT=R/'s6_ready_v1';OUT.mkdir(exist_ok=False)
ref=Path('/data_cf/cowmpo/worlds/s4_ref_goal_v02')
refout=OUT/'reference_world_pair';shutil.copytree(ref,refout)
w=json.loads((refout/'world.json').read_text());w.pop('world_sha256');w['table']['contact_rule']='object_table_pair'
w['contact_ablation_note']='frozen reference assets; only contact mixing changes'
w['world_sha256']=hashlib.sha256(json.dumps(w,sort_keys=True).encode()).hexdigest()
(refout/'world.json').write_text(json.dumps(w,indent=2))
shape=R/'world_pair_v1';shape_meta=json.loads((shape/'world.json').read_text());shape_hash=tree_sha(shape)
records=[]
for dataset,source,evidence,state_root,probe_root in [
 ('dev3',R/'visual_dev_bound_v1/dev3',Path('/data_cf/cowmpo/evidence/libero_goal_dev3'),
  R/'state_visual_dev_merged_v1/dev3',Path('/data_cf/cowmpo/worlds/s6_strike/dev3')),
 ('fit3-dev',R/'visual_dev_bound_v1/fit3',Path('/data_cf/cowmpo/evidence/libero_goal_fit3'),
  R/'state_visual_dev_merged_v1/fit3',Path('/data_cf/cowmpo/worlds/s6_fit3/probes')),
 ('fit3',R/'visual_fit_result_v3/fit3',Path('/data_cf/cowmpo/evidence/libero_goal_fit3'),
  R/'state_fit_visual_v3',Path('/data_cf/cowmpo/worlds/s6_fit3/probes'))]:
    for ep in sorted(p.name for p in source.iterdir() if p.is_dir()):
        role=roles.role(ep);roles.require([ep],{'dev','fit'},'S6 preparation')
        dest=OUT/'anchors'/role/ep;dest.mkdir(parents=True)
        for name in ['masks.npz','association.json']:shutil.copy2(source/ep/name,dest/name)
        ed=OUT/'evidence'/ep;ed.mkdir(parents=True);shutil.copy2(evidence/ep/'frame0.png',ed/'frame0.png')
        sd=OUT/'shape_states'/ep;sd.mkdir(parents=True)
        path=state_root/ep/'state.json';state=json.loads(path.read_text())
        state.update(world_sha256=shape_meta['world_sha256'],world_tree_sha256=shape_hash,
           derivation={'change':'contact mixing only; poses and geometry retained',
                       'source_state_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        (sd/'state.json').write_text(json.dumps(state,indent=1))
        pd=OUT/'probes'/ep;pd.mkdir(parents=True);shutil.copy2(probe_root/ep/'probe.json',pd/'probe.json')
        records.append({'episode':ep,'role':role,'dataset':dataset})
assert sum(r['role']=='fit' for r in records)==28 and sum(r['role']=='dev' for r in records)==32
(OUT/'inputs.json').write_text(json.dumps({'episodes':records,'reference_hash':tree_sha(refout),
    'shape_world':str(shape),'shape_hash':shape_hash,'gate_test_queried':False},indent=2))
print('S6_INPUTS_READY',len(records))
