# Copyright 2026 The RLinf Authors.
"""Target-private single-factor attribution of the frozen orthogonal rack responses."""
import json
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene,iou
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');EP='libero_goal_t0_i0';keys=json.loads((R/'rack_multidir_preflight_dev_v1/selection.json').read_text())['selected'];scene=Scene(str(R/'bowl_support_dev_v1/world11'))
robot=json.loads((R/'robot_rgb_all_dev_v2'/EP/'robot.json').read_text())['robot']
with np.load(T/'robot_initial_dev_v1'/(EP+'.npz')) as a:robot.update(arm_qpos=a['arm_q'].tolist(),grip_qpos=a['grip_q'].tolist(),arm_qvel=a['arm_qvel'].tolist(),grip_qvel=a['grip_qvel'].tolist(),grip_current_action=a['grip_current_action'].tolist())
state=json.loads((R/'natural_states_all_dev_v1'/EP/'state.json').read_text())['objects'];rows=[]
for key in keys:
    scene.reset();apply_state(scene,state,require_matched=True,robot=robot);probe=json.loads((R/'rack_multidir_preflight_dev_v1'/key/'probe.json').read_text());values=[]
    for step,act in enumerate(probe['actions'],1):
        scene.env.step(np.asarray(act))
        if 190<=step<=300 and step%10==0:
            with np.load(R/'rack_multidir_video_cpu_v1'/key/f'{EP}__f{step:03d}'/'masks.npz') as a:target=a['robot']
            values.append(iou(scene.robot_mask(),target))
    rows.append({'query':key,'mean_iou':float(np.mean(values))})
scene.env.close();report={'episode':EP,'oracle_robot_response_iou':float(np.mean([r['mean_iou'] for r in rows])),'estimated_robot_response_iou':json.loads((R/'rack_multidir_candidates_dev_v1/candidate06.json').read_text())['robot_iou'],'rows':rows,'scope':'Target-private robot-only replacement; geometry and objects unchanged; no oracle record exported'}
(T/'rack_robot_attribution_v1.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)
