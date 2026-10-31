# Copyright 2026 The RLinf Authors.
"""Export episode robot initialization into Target-private storage for diagnosis only."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm import roles
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/robot_initial_dev_v1')
def worker(ep):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    roles.require([ep],{'dev'},'Target robot initialization diagnosis')
    root=Path('/data_cf/cowmpo/init')/('libero_goal_dev3' if roles.parse(ep)[1] in [0,1,2] else 'libero_goal_fit3')
    manifest=json.loads((root/'manifest.json').read_text());info=manifest['episodes'][ep+'.npy'];suite=benchmark.get_benchmark_dict()[manifest['suite']]();task=suite.get_task(info['task_id'])
    env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False)
    env.reset();sim=env.env.sim;priv=np.load(root/info['privileged']);sim.model.body_pos[:]=priv['body_pos'];sim.model.body_quat[:]=priv['body_quat']
    env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']]);warm=np.zeros(7);warm[6]=-1
    for _ in range(manifest['num_steps_wait']):env.step(warm)
    r=env.env.robots[0];obs=env.env._get_observations(force_update=True)
    np.savez_compressed(T/(ep+'.npz'),arm_q=sim.data.qpos[r._ref_joint_pos_indexes],grip_q=sim.data.qpos[r._ref_gripper_joint_pos_indexes],arm_qvel=sim.data.qvel[r._ref_joint_vel_indexes],grip_qvel=sim.data.qvel[r._ref_gripper_joint_vel_indexes],grip_current_action=r.gripper.current_action,eef=obs['robot0_eef_pos'])
    env.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--episode');a=p.parse_args()
    if a.episode:return worker(a.episode)
    T.mkdir(exist_ok=False);eps=sorted(p.parent.name for p in (R/'stove_parts_states_dev_v1').glob('*/state.json'))
    def run(ep):
        with (T/(ep+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--episode',ep],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((ep,r.returncode))
    with ThreadPoolExecutor(8) as pool:list(pool.map(run,eps))
    print('ROBOT_INITIAL_TARGET_PRIVATE_COMPLETE',len(eps),flush=True)
if __name__=='__main__':main()
