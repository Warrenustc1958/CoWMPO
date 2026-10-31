"""Target-only residual attribution of the frozen remaining-dev grasp experiment."""
import json
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from replay_rgb_robot import install_robot
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
score=json.loads((T/'grasp_natural_remaining_dev_v1.json').read_text());wrong={k.split('__')[0] for k in score['wrong']};scene=Scene(str(R/'bowl_support_dev_v1/world11'));rows=[]
for ep in sorted(wrong):
    scene.reset();doc=json.loads((R/'natural_states_all_dev_v1'/ep/'state.json').read_text());apply_state(scene,doc['objects'],require_matched=True);install_robot(scene,json.loads((R/'robot_rgb_all_dev_v1'/ep/'robot.json').read_text())['robot'])
    with np.load(T/'robot_initial_dev_v1'/(ep+'.npz')) as a:robot_error=float(np.linalg.norm(scene.env.eef_pos()-a['eef'])*1000)
    data='dev3' if ep.rsplit('_i',1)[1] in ['0','1','2'] else 'fit3';gt=json.loads((Path('/data_cf/cowmpo/target_eval/s5_gt')/data/(ep+'.json')).read_text());errors={}
    for n in ['cream_cheese','bowl']:
        s=doc['objects'][n];errors[n]={'xy_mm':float(np.linalg.norm(np.asarray(s['aabb_center_rel_base'][:2])-gt[n]['aabb_center_rel_base'][:2])*1000),'z_mm':float(abs(s['aabb_center_rel_base'][2]-gt[n]['aabb_center_rel_base'][2])*1000),'extent_max_mm':float(np.max(np.abs(np.asarray(s['aabb_half'])-gt[n]['aabb_half']))*2000)}
    rows.append({'episode':ep,'robot_eef_mm':robot_error,'object_error':errors})
scene.env.close();oracle_wrong=[]
for key in json.loads((R/'grasp_contrast_dev_holdout_v1/freeze.json').read_text())['episodes']:
    for mode in ['grasp','miss']:
        k=key+'__'+mode;label=json.loads((T/'grasp_contrast_dev_holdout_v1'/(k+'.json')).read_text())['label'];pred=json.loads((T/'grasp_oracle_robot_shape_dev_v1'/(k+'.json')).read_text())['grasped']
        if label!=pred:oracle_wrong.append(k)
report={'residual_rows':rows,'old_robot_plus_cheese_oracle_wrong':oracle_wrong,'scope':'Target-only attribution; numerical GT values never become Builder inputs'}
(T/'grasp_residual_diagnostic_v1.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
