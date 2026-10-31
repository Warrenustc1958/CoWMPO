"""Target-private platform parity and object contact attribution; exports no object parameters."""
import json
from pathlib import Path
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from dump_builder_platform import platform
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
suite=benchmark.get_benchmark_dict()['libero_goal']();task=suite.get_task(0)
env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False)
env.reset();m=env.env.sim.model;target=platform(m);source=json.loads((R/'builder_platform_v1.json').read_text());differences=[]
for name in sorted(source.keys()&target.keys()):
    for field,value in source[name].items():
        a,b=np.asarray(value),np.asarray(target[name][field])
        if a.shape!=b.shape or not np.allclose(a,b,atol=1e-9,rtol=1e-8):differences.append({'component':name,'field':field,'shape_equal':a.shape==b.shape})
report={'builder_components':len(source),'target_components':len(target),'missing_in_target':sorted(source.keys()-target.keys()),'missing_in_builder':sorted(target.keys()-source.keys()),'different_fields':differences,'scope':'shared platform physical fields only; not pose or controller cache'}
(T/'platform_parity_v1.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
root=next(b for b in range(m.nbody) if int(m.body_rootid[b])==b and (m.body_id2name(b) or '').startswith('cream_cheese_'))
gids=[g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]])==root and m.geom_contype[g]!=0]
if not gids:raise ValueError('no cheese collision geometry')
data={n:getattr(m,'geom_'+n)[gids[0]].tolist() for n in ['friction','solref','solimp','condim','margin','gap']}
data.update(mass=float(m.body_mass[root]),inertia=m.body_inertia[root].tolist(),iquat=m.body_iquat[root].tolist())
(T/'cheese_contact_private_v1.json').write_text(json.dumps(data,indent=1));env.close()
