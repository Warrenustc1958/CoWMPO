# Copyright 2026 The RLinf Authors.
"""Three-height rack contact queries from Builder poses; evidence is RGB only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from cowm import roles
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'rack_touch_dev_v1'
EP='libero_goal_t0_i0';roles.require([EP],{'dev'},'rack touch development')
WORLD=R/'stove_parts_geometry_dev_v1/world07';SOURCE=R/'stove_parts_states_dev_v1'/EP/'state.json'
OUT.mkdir(exist_ok=False);scene=Scene(str(WORLD));state=json.loads(SOURCE.read_text())['objects']
rx,ry=state['rack']['pos'][:2];top=scene.world['table']['height']
for height in [.07,.17,.27]:
    key=f'h{round(height*1000):03d}';root=OUT/key;probe=root/'probes'/EP;probe.mkdir(parents=True)
    evidence=root/'predicted_frames';evidence.mkdir();scene.reset();apply_state(scene,state,require_matched=True)
    acts=[];contacts=[]
    def save(step):
        dest=evidence/f'{EP}__f{step:03d}';dest.mkdir()
        Image.fromarray(scene.rgb()).save(dest/'frame0.png')
        np.savez_compressed(dest/'masks.npz',robot=scene.robot_mask())
    save(0)
    def advance(action):
        scene.env.step(np.asarray(action));acts.append(list(map(float,action)))
        sim=scene.env.sim;k=scene.names.index('rack')
        hit=any((scene.geom2obj[int(c.geom1)]==k and scene.robot_geom[int(c.geom2)]) or
                (scene.geom2obj[int(c.geom2)]==k and scene.robot_geom[int(c.geom1)])
                for c in sim.data.contact[:sim.data.ncon])
        contacts.append(bool(hit))
        if len(acts)%10==0:save(len(acts))
    def drive(target,n):
        for _ in range(n):
            action=np.zeros(7);action[:3]=np.clip(10*(np.asarray(target)-scene.env.eef_pos()),-1,1);action[6]=1
            advance(action)
    drive([rx-.20,ry,top+.39],60)
    drive([rx-.20,ry,top+height],60)
    drive([rx+.08,ry,top+height],100)
    drive([scene.env.eef_pos()[0],ry,top+.39],50)
    frames=list(range(0,len(acts)+1,10))
    rec={'episode':EP,'obj':'rack','mode':'touch_x','height_above_table':height,'actions':acts,'frames':frames,
       'world_tree_sha256':tree_sha(WORLD),'state_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
       'builder_predicted_contact_steps':[i+1 for i,v in enumerate(contacts) if v],
       'purpose':'constrain rack support geometry from robot response in RGB; no Target state feedback'}
    (probe/'probe.json').write_text(json.dumps(rec,indent=1))
    print('rack_touch_planned',key,len(acts),sum(contacts),flush=True)
scene.env.close()
