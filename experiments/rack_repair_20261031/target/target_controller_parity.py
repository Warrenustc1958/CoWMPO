# Copyright 2026 The RLinf Authors.
"""Target-private controller parity attribution with identical robot states and actions."""
import argparse,json
from pathlib import Path
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');OUT=T/'controller_parity_v1';OUT.mkdir(exist_ok=False)
rows=[]
for ep in ['libero_goal_t0_i0','libero_goal_t1_i5']:
    init=Path('/data_cf/cowmpo/init')/('libero_goal_fit3' if ep.endswith('_i5') else 'libero_goal_dev3');manifest=json.loads((init/'manifest.json').read_text());info=manifest['episodes'][ep+'.npy'];suite=benchmark.get_benchmark_dict()[manifest['suite']]();task=suite.get_task(info['task_id'])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True);env.reset();sim=env.env.sim;m=sim.model
    with np.load(init/info['privileged']) as priv:m.body_pos[:]=priv['body_pos'];m.body_quat[:]=priv['body_quat']
    env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']]);warm=np.zeros(7);warm[6]=-1
    for _ in range(manifest['num_steps_wait']):env.step(warm)
    target=env.env.robots[0];s={'arm_qpos':sim.data.qpos[target._ref_joint_pos_indexes].tolist(),'arm_qvel':sim.data.qvel[target._ref_joint_vel_indexes].tolist(),'grip_qpos':sim.data.qpos[target._ref_gripper_joint_pos_indexes].tolist(),'grip_qvel':sim.data.qvel[target._ref_gripper_joint_vel_indexes].tolist(),'grip_current_action':target.gripper.current_action.tolist()}
    scene=Scene(str(R/'bowl_support_dev_v1/world11'));scene.reset();apply_state(scene,json.loads((R/'natural_states_all_dev_v1'/ep/'state.json').read_text())['objects'],require_matched=True,robot=s);code=scene.env.robots[0]
    target.controller.update(force=True);code.controller.update(force=True);compared={};different={}
    for key,value in vars(target.controller).items():
        try:a=np.asarray(value);b=np.asarray(getattr(code.controller,key))
        except (ValueError,TypeError,AttributeError):continue
        if a.dtype.kind not in 'fiub' or a.shape!=b.shape or a.size==0:continue
        error=float(np.max(np.abs(a.astype(float)-b.astype(float))));compared[key]=error
        if error>1e-10:different[key]=error
    opts={}
    for key in ['timestep','iterations','ls_iterations','tolerance','ls_tolerance','impratio','gravity','solver','integrator','cone','enableflags','disableflags','jacobian']:
        if hasattr(sim.model.opt,key) and hasattr(scene.env.sim.model.opt,key):
            a=np.asarray(getattr(sim.model.opt,key));b=np.asarray(getattr(scene.env.sim.model.opt,key));diff=float(np.max(np.abs(a.astype(float)-b.astype(float))))
            if diff:opts[key]=diff
    source=R/('grasp_contrast_dev_holdout_v1' if ep.endswith('_i5') else 'grasp_contrast_dev_v1')/(ep+'__grasp')/'probe.json';probe=json.loads(source.read_text());trajectory=[]
    for step,action in enumerate(probe['actions'][:30],1):
        env.step(np.asarray(action));scene.env.step(np.asarray(action));qerr=float(np.max(np.abs(sim.data.qpos[target._ref_joint_pos_indexes]-scene.env.sim.data.qpos[code._ref_joint_pos_indexes])));eef=float(np.linalg.norm(target._hand_pos-scene.env.eef_pos())*1000)
        trajectory.append({'step':step,'max_arm_qpos_error':qerr,'eef_mm':eef})
    row={'episode':ep,'compared_controller_fields':len(compared),'different_controller_fields':different,'different_options':opts,'first30':trajectory};rows.append(row);scene.env.close();env.close()
(OUT/'report.json').write_text(json.dumps(rows,indent=1));print(json.dumps([{'episode':r['episode'],'compared_controller_fields':r['compared_controller_fields'],'different_controller_fields':r['different_controller_fields'],'different_options':r['different_options'],'last':r['first30'][-1]} for r in rows],indent=1))
