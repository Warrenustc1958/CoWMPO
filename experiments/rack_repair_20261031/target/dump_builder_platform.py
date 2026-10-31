"""Record code-world robot dynamics for an isolated platform parity audit."""
import json
from pathlib import Path
from cowm.init.estimate_state import Scene
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
def platform(m):
    out={}
    for kind,fields in [('geom',['size','friction','solref','solimp','condim','contype','conaffinity']),('body',['mass','inertia','ipos','iquat']),('joint',['stiffness','range']),('actuator',['gainprm','biasprm','ctrlrange','forcerange','gear'])]:
        count=getattr(m,'n'+('jnt' if kind=='joint' else ('u' if kind=='actuator' else kind)))
        for i in range(count):
            name=getattr(m,kind+'_id2name')(i) or ''
            if not name.startswith(('robot0','gripper0')):continue
            row={}
            for field in fields:
                attr=('jnt_' if kind=='joint' else kind+'_')+field
                x=getattr(m,attr)[i];row[field]=x.tolist() if hasattr(x,'tolist') else x
            out[kind+'/'+name]=row
    return out
if __name__=='__main__':
    scene=Scene(str(R/'stove_parts_geometry_dev_v1/world07'));scene.reset();out=platform(scene.env.sim.model)
    (R/'builder_platform_v1.json').write_text(json.dumps(out,indent=1));scene.env.close();print('BUILDER_PLATFORM_EXPORTED',len(out))
