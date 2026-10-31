# Copyright 2026 The RLinf Authors.
"""Target-private shared-platform geometry audit; no scene assets or parameters exported."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');OUT=T/'robot_platform_audit_v2';OUT.mkdir(exist_ok=False);EP='libero_goal_t0_i0'
def platform(m,root):
    out={}
    for g in range(m.ngeom):
        body=m.body_id2name(int(m.geom_bodyid[g])) or ''
        if not (int(m.body_rootid[m.geom_bodyid[g]])==m.body_name2id(root) or body.startswith(('robot0','gripper0'))):continue
        name=m.geom_id2name(g) or ('unnamed_'+str(g));rec={'body':body}
        for key in ['geom_type','geom_group','geom_size','geom_pos','geom_quat','geom_rgba','geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp']:
            rec[key]=np.asarray(getattr(m,key)[g]).tolist()
        mid=int(m.geom_dataid[g]);rec['mesh']=None
        if int(m.geom_type[g])==7 and mid>=0:
            adr=int(m.mesh_vertadr[mid]);num=int(m.mesh_vertnum[mid]);verts=m.mesh_vert[adr:adr+num];rec['mesh']={'vertices':num,'sha256':hashlib.sha256(verts.tobytes()).hexdigest(),'extent':np.ptp(verts,axis=0).tolist()}
        out[name]=rec
    return out
init=Path('/data_cf/cowmpo/init/libero_goal_dev3');man=json.loads((init/'manifest.json').read_text());info=man['episodes'][EP+'.npy'];suite=benchmark.get_benchmark_dict()[man['suite']]();task=suite.get_task(info['task_id']);env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True);env.reset();sim=env.env.sim;m=sim.model
with np.load(init/info['privileged']) as a:m.body_pos[:]=a['body_pos'];m.body_quat[:]=a['body_quat']
obs=env.set_init_state(suite.get_task_init_states(info['task_id'])[info['trial_id']]);warm=np.zeros(7);warm[6]=-1
for _ in range(man['num_steps_wait']):obs,_,_,_=env.step(warm)
target=platform(m,env.env.robots[0].robot_model.root_body);rroot=m.body_name2id(env.env.robots[0].robot_model.root_body);target_group=sim._render_context_offscreen.vopt.geomgroup.copy().tolist();rgb=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);raw=sim.render(width=256,height=256,camera_name='agentview',segmentation=True);gid=raw[...,1].astype(int);gid[(raw[...,0]!=5)|(gid<0)]=-1;gid=gid[::-1,::-1];robgeom=np.array([int(m.body_rootid[m.geom_bodyid[g]])==rroot or (m.body_id2name(int(m.geom_bodyid[g])) or '').startswith(('robot0','gripper0')) for g in range(m.ngeom)]);targetmask=(gid>=0)&robgeom[np.clip(gid,0,None)];env.close()
scene=Scene(str(R/'bowl_support_dev_v1/world11'));scene.reset();record=json.loads((R/'robot_rgb_all_dev_v2'/EP/'robot.json').read_text())['robot']
with np.load(T/'robot_initial_dev_v1'/(EP+'.npz')) as a:record.update(arm_qpos=a['arm_q'].tolist(),grip_qpos=a['grip_q'].tolist(),arm_qvel=a['arm_qvel'].tolist(),grip_qvel=a['grip_qvel'].tolist(),grip_current_action=a['grip_current_action'].tolist())
apply_state(scene,json.loads((R/'natural_states_all_dev_v1'/EP/'state.json').read_text())['objects'],require_matched=True,robot=record);code=platform(scene.env.sim.model,scene.env.robots[0].robot_model.root_body);codemask=scene.robot_mask();coderender=scene.rgb();code_group=scene.env.sim._render_context_offscreen.vopt.geomgroup.copy().tolist();scene.env.close()
diff=[]
for name in sorted(set(target)&set(code)):
    changed=[k for k in target[name] if target[name][k]!=code[name][k]]
    if changed:diff.append({'geom':name,'fields':changed})
report={'target_robot_geoms':len(target),'code_robot_geoms':len(code),'only_target':sorted(set(target)-set(code)),'only_code':sorted(set(code)-set(target)),'changed':diff,'target_render_groups':target_group,'code_render_groups':code_group,'target_pixels':int(targetmask.sum()),'code_pixels':int(codemask.sum()),'iou':float((targetmask&codemask).sum()/(targetmask|codemask).sum()),'scope':'shared robot platform only; detailed source metadata stays Target-private'}
(OUT/'private_platforms.json').write_text(json.dumps({'Target':target,'Code':code},indent=1));(OUT/'summary.json').write_text(json.dumps(report,indent=1))
overlay=np.zeros((256,256,3),np.uint8);overlay[...,0]=targetmask*255;overlay[...,1]=codemask*255;Image.fromarray(np.concatenate([rgb,coderender,overlay],axis=1)).save(OUT/'comparison.png');print(json.dumps(report,indent=1))
