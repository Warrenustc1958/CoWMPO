"""Target-only diagnosis: conservative rotated geom boxes versus actual mesh bounds. Dev only."""
import json
from pathlib import Path

import numpy as np

from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from cowm.init.aabb import object_aabb

root=Path('/data_cf/cowmpo/init/libero_goal_dev3')
manifest=json.loads((root/'manifest.json').read_text())
entry=manifest['episodes']['libero_goal_t0_i0.npy']
suite=benchmark.get_benchmark_dict()[manifest['suite']]()
task=suite.get_task(entry['task_id'])
env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                       camera_heights=256,camera_widths=256,ignore_done=True)
env.reset();priv=np.load(root/entry['privileged']);sim=env.env.sim
sim.model.body_pos[:]=priv['body_pos'];sim.model.body_quat[:]=priv['body_quat']
env.set_init_state(suite.get_task_init_states(entry['task_id'])[entry['trial_id']])
warm=np.zeros(7);warm[6]=-1
for _ in range(manifest['num_steps_wait']):env.step(warm)
m,d=sim.model,sim.data
out={}
for label,prefix in [('stove','flat_stove'),('rack','wine_rack')]:
    roots=[b for b in range(m.nbody) if int(m.body_rootid[b])==b and
           (m.body_id2name(b) or '').startswith(prefix+'_')]
    gids=[g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]])==roots[0]]
    lo,hi=np.full(3,np.inf),np.full(3,-np.inf);rows=[]
    for g in gids:
        R=d.geom_xmat[g].reshape(3,3);p=d.geom_xpos[g];kind=int(m.geom_type[g]);s=m.geom_size[g]
        if kind==7:
            mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
            pts=m.mesh_vert[start:start+n]@R.T+p
            l,h=pts.min(0),pts.max(0)
        else:
            a=np.asarray(m.geom_aabb[g]);extent=np.abs(R)@a[3:];c=p+R@a[:3];l,h=c-extent,c+extent
        lo=np.minimum(lo,l);hi=np.maximum(hi,h)
        a=np.asarray(m.geom_aabb[g]);ext=np.abs(R)@a[3:];center=p+R@a[:3]
        rows.append({'name':m.geom_id2name(g),'kind':kind,'group':int(m.geom_group[g]),
                     'old_lo':(center-ext).tolist(),'old_hi':(center+ext).tolist(),
                     'exact_lo':l.tolist(),'exact_hi':h.tolist()})
    center,half=object_aabb(m,d,gids)
    out[label]={'old_center':center.tolist(),'old_half':half.tolist(),
                'mesh_exact_center':((lo+hi)/2).tolist(),'mesh_exact_half':((hi-lo)/2).tolist(),'geoms':rows}
env.close()
path=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/aabb_diagnostic.json')
path.write_text(json.dumps(out,indent=1))
for n,v in out.items():print(n,{k:x for k,x in v.items() if k!='geoms'})
