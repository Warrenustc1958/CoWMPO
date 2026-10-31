"""Target-only evaluation of image-estimated robot initialization."""
import json
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
scene=Scene(str(R/'stove_parts_geometry_dev_v1/world07'));rows=[]
for file in sorted((R/'robot_rgb_dev_v1').glob('*/robot.json')):
    ep=file.parent.name;fit=json.loads(file.read_text());scene.reset();apply_state(scene,json.loads((R/'stove_parts_states_dev_v1'/ep/'state.json').read_text())['objects'],require_matched=True)
    r=scene.env.robots[0];sim=scene.env.sim
    with np.load(T/'robot_initial_dev_v1'/(ep+'.npz')) as a:gt={k:a[k] for k in a.files}
    nominal=float(np.linalg.norm(scene.env.eef_pos()-gt['eef'])*1000)
    sim.data.qpos[r._ref_joint_pos_indexes]=fit['robot']['arm_qpos'];sim.forward()
    error=float(np.linalg.norm(scene.env.eef_pos()-gt['eef'])*1000)
    rows.append({'episode':ep,'nominal_eef_mm':nominal,'rgb_fitted_eef_mm':error,'delta_norm_rad':fit['delta_norm_rad'],'rgb_rmse_before':fit['initial_rgb_rmse'],'rgb_rmse_after':fit['fitted_rgb_rmse']})
scene.env.close();(T/'robot_rgb_dev_v1.json').write_text(json.dumps(rows,indent=1));print(json.dumps(rows,indent=1))
