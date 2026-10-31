# Copyright 2026 The RLinf Authors.
"""Target-private evaluation of initial-RGB robot estimates on six development episodes."""
import json
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
scene=Scene(str(R/'bowl_support_dev_v1/world11'));rows=[]
for k in range(6):
    ep=f'libero_goal_t{k}_i0';row={'episode':ep}
    with np.load(T/'robot_initial_dev_v1'/(ep+'.npz')) as a:gt={n:a[n] for n in a.files}
    for name,root in [('baseline','robot_rgb_all_dev_v2'),('affine','robot_rgb_affine_dev_v1')]:
        record=json.loads((R/root/ep/'robot.json').read_text())['robot'];scene.reset()
        apply_state(scene,json.loads((R/'natural_states_all_dev_v1'/ep/'state.json').read_text())['objects'],require_matched=True,robot=record)
        row[name]={'eef_error_mm':float(np.linalg.norm(scene.env.eef_pos()-gt['eef'])*1000),
                   'finger_q_error_mm':float(np.linalg.norm(np.asarray(record['grip_qpos'])-gt['grip_q'])*1000),
                   'arm_q_error_rad':float(np.linalg.norm(np.asarray(record['arm_qpos'])-gt['arm_q']))}
    rows.append(row)
scene.env.close();report={'episodes':6,'rows':rows,'scope':'Target-private diagnostic only; GT never used for robot fitting'}
(T/'robot_rgb_affine_dev_v1.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
