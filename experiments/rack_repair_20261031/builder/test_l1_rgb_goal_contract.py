# Copyright 2026 The RLinf Authors.
"""Real L1 initialization, generate and rejection contracts on frozen RGB-derived states."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SC=Path(__file__).parent;OUT=R/'l1_rgb_contract_v2';OUT.mkdir(exist_ok=False)
REPO='/mnt/nfs/fjj/Code/CoWMPO';sys.path.insert(1,REPO)
import rlinf.envs.sim.world_model.backend
name='rlinf.envs.sim.world_model.backend.code';spec=importlib.util.spec_from_file_location(name,SC/'rlinf/envs/sim/world_model/backend/code.py');module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
be=object.__new__(module.CodeBackend);be.world_dir=str(R/'bowl_support_dev_v1/world11');be.state_dir=str(R/'s6_natural_v2/shape_states');be.world_friction={};be._envs={};be._first={};be._open={0};be.allow_legacy_state=False
EP='libero_goal_t0_i0';doc=json.loads((Path(be.state_dir)/EP/'state.json').read_text());be._open_l1(0,EP)
reference=Scene(be.world_dir);reference.reset();apply_state(reference,doc['objects'],require_matched=True,robot=doc['robot'])
np.testing.assert_array_equal(be._envs[0]['env'].sim.data.qpos,reference.env.sim.data.qpos)
np.testing.assert_array_equal(be._envs[0]['env'].sim.data.qvel,reference.env.sim.data.qvel)
np.testing.assert_array_equal(be._first[0].numpy(),torch.from_numpy(reference.rgb().astype(np.float32)/255*2-1).permute(2,0,1).numpy())
actions=np.random.default_rng(64).uniform(-.25,.25,(1,8,7)).astype(np.float32);actions[:,:,6]=-1
chunk=be.generate([0],torch.from_numpy(actions));qpos=be._envs[0]['env'].sim.data.qpos.copy();qvel=be._envs[0]['env'].sim.data.qvel.copy()
frames=[]
for action in actions[0]:
 obs,_,_,_=reference.env.step(action);frames.append(module.CodeBackend.frame_from_obs(obs))
np.testing.assert_array_equal(chunk.numpy(),torch.stack(frames,dim=1)[None].numpy());np.testing.assert_array_equal(qpos,reference.env.sim.data.qpos);np.testing.assert_array_equal(qvel,reference.env.sim.data.qvel)
be._open_l1(0,EP);single=torch.cat([be.generate([0],torch.from_numpy(actions[:,k:k+1])) for k in range(8)],dim=2);np.testing.assert_array_equal(single.numpy(),chunk.numpy());np.testing.assert_array_equal(be._envs[0]['env'].sim.data.qpos,qpos)
checks={'installed_qpos_qvel_rgb':True,'generate_vs_direct_bitwise':True,'chunk_vs_single_bitwise':True}
root=OUT/'bad_states';p=root/EP;p.mkdir(parents=True);be.state_dir=str(root)
for condition in ['ambiguous','missing_robot','bad_integrator','wrong_tree_hash','wrong_episode','legacy_default']:
 bad=json.loads(json.dumps(doc))
 if condition=='ambiguous':bad['objects']['cream_cheese']['association_status']='ambiguous'
 elif condition=='missing_robot':bad.pop('robot')
 elif condition=='bad_integrator':bad['robot']['grip_current_action']=[-1.]
 elif condition=='wrong_tree_hash':bad['world_tree_sha256']='0'*64
 elif condition=='wrong_episode':bad['episode']='libero_goal_t1_i0'
 else:bad.pop('world_tree_sha256');bad.pop('episode')
 (p/'state.json').write_text(json.dumps(bad));before=be._envs[0]['env'].sim.data.qpos.copy()
 try:be._open_l1(0,EP)
 except ValueError:checks[condition+'_rejected']=True
 else:raise AssertionError('L1 accepted '+condition)
 np.testing.assert_array_equal(be._envs[0]['env'].sim.data.qpos,before)
reference.env.close()
for v in be._envs.values():v['env'].close()
(OUT/'result.json').write_text(json.dumps(checks,indent=1));print('L1_RGB_CONTRACT_PASS',json.dumps(checks),flush=True)
