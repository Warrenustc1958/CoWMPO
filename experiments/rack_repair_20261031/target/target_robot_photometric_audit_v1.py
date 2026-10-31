# Copyright 2026 The RLinf Authors.
"""Diagnose whether the current RGB objective favors a wrong robot pose, Target-private."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion,gaussian_filter
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');scene=Scene(str(R/'bowl_support_dev_v1/world11'));rows=[]
for k in range(6):
    ep=f'libero_goal_t{k}_i0';state=json.loads((R/'natural_states_all_dev_v1'/ep/'state.json').read_text())['objects'];scene.reset();apply_state(scene,state,require_matched=True)
    roi=binary_dilation(scene.robot_mask(),iterations=5);interior=binary_erosion(scene.robot_mask(),iterations=2)
    with np.load(R/'visual_dev_bound_v1/dev3'/ep/'masks.npz') as a:occupied=np.logical_or.reduce([a[n] for n in a.files])
    roi&=~binary_dilation(occupied,iterations=2);interior&=roi
    image=Path('/data_cf/cowmpo/evidence/libero_goal_dev3')/ep/'frame0.png';target=gaussian_filter(np.asarray(Image.open(image).convert('RGB'),dtype=float)/255,(1.2,1.2,0));row={'episode':ep}
    for name in ['rgb_estimated','Target_oracle']:
        robot=json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot']
        if name=='Target_oracle':
            with np.load(T/'robot_initial_dev_v1'/(ep+'.npz')) as a:
                robot.update(arm_qpos=a['arm_q'].tolist(),grip_qpos=a['grip_q'].tolist(),arm_qvel=a['arm_qvel'].tolist(),grip_qvel=a['grip_qvel'].tolist(),grip_current_action=a['grip_current_action'].tolist())
        scene.reset();apply_state(scene,state,require_matched=True,robot=robot);render=gaussian_filter(np.asarray(scene.rgb(),dtype=float)/255,(1.2,1.2,0));res=render-target
        row[name]={'roi_rgb_rmse':float(np.sqrt(np.mean(res[roi]**2))),'interior_rgb_rmse':float(np.sqrt(np.mean(res[interior]**2)))}
    rows.append(row)
scene.env.close();report={'rows':rows,'oracle_loss_above_estimated':sum(r['Target_oracle']['roi_rgb_rmse']>r['rgb_estimated']['roi_rgb_rmse'] for r in rows),'scope':'Target-private objective diagnostic; no oracle records exported to Builder'}
(T/'robot_photometric_audit_v1.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
