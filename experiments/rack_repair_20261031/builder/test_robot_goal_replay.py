# Copyright 2026 The RLinf Authors.
"""Real CPU contract: G2 must replay the provided robot initialization and controller state."""
import json
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.init.robot_state import install_robot_state
from cowm.s6.plan_probe import apply_state
from cowm.s6.g2_replay import replay
import cowm
root=Path(__file__).parent
if Path(cowm.__file__).resolve()!=root/'toolkits/cowm/__init__.py':raise RuntimeError('unexpected package path')
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');ep='robot_contract';out=R/'robot_replay_contract_v3';out.mkdir(exist_ok=False)
scene=Scene(str(R/'bowl_support_dev_v1/world11'));scene.reset()
objects=json.loads((R/'bowl_support_dev_v1/state11/libero_goal_t0_i0/state.json').read_text())['objects'];robot=json.loads((R/'robot_rgb_all_dev_v2/libero_goal_t0_i0/robot.json').read_text())['robot']
actions=[[.1,0,0,0,0,0,-1.],[0,.1,0,0,0,0,-1.],[0,0,.1,0,0,0,-1.]]
orientation_prior=scene.env.robots[0].controller.goal_ori.copy();apply_state(scene,objects,require_matched=True,robot=robot);np.testing.assert_array_equal(scene.env.robots[0].controller.goal_ori,orientation_prior)
if np.max(np.abs(scene.env.robots[0].controller.ee_ori_mat-orientation_prior))<1e-8:raise AssertionError('fixture must distinguish current orientation from reset goal')
k=scene.names.index('cream_cheese')
def save(step):
    dest=out/f'{ep}__f{step:03d}';dest.mkdir();np.savez_compressed(dest/'masks.npz',cream_cheese=scene.seg()==k)
save(0)
for step,action in enumerate(actions,1):scene.env.step(np.asarray(action));save(step)
expected_qpos=scene.env.sim.data.qpos.copy();expected_qvel=scene.env.sim.data.qvel.copy();expected_rgb=scene.rgb().copy()
probe={'episode':ep,'obj':'cream_cheese','actions':actions,'frames':[0,1,2,3]};validity={'obj':'cream_cheese','terminal_step':3,'frames_planned':[0,1,2,3],'frames':{str(i):{'cream_cheese':'valid'} for i in range(4)}}
result=replay(scene,objects,probe,str(out),validity,robot=robot)
np.testing.assert_array_equal(scene.env.sim.data.qpos,expected_qpos);np.testing.assert_array_equal(scene.env.sim.data.qvel,expected_qvel);np.testing.assert_array_equal(scene.rgb(),expected_rgb)
replay(scene,objects,probe,str(out),validity)
if np.array_equal(scene.env.sim.data.qpos,expected_qpos):raise AssertionError('robot initialization had no observable effect')
scene.env.close();(out/'result.json').write_text(json.dumps({'qpos':True,'qvel':True,'RGB':True,'fixed_reset_control_differs':True,'scope':'mechanical interface self replay only, not Target dynamics validation'}));print('ROBOT_G2_REPLAY_CONTRACT_PASS qpos/qvel/RGB and control',flush=True)
