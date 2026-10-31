# Copyright 2026 The RLinf Authors.
"""Track shared robot parts independently when the arm leaves the camera frustum."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import binary_dilation
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SC=Path(__file__).parent;OUT=R/'rack_robot_parts_cpu_v4';EP='libero_goal_t0_i0'
SETS=[('horizontal','rack_isolated_preflight_dev_v2','codex_rack_isolated_dev_v2'),('vertical','rack_vertical_preflight_dev_v3','codex_rack_vertical_dev_v3')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    from cowm.init.estimate_state import Scene
    from cowm.s6.plan_probe import apply_state
    from cowm.world.artifact import tree_sha
    OUT.mkdir(exist_ok=False);world=R/'bowl_support_dev_v1/world11';state=R/'natural_states_all_dev_v1'/EP/'state.json';robot=R/'robot_rgb_all_dev_v2'/EP/'robot.json';source=R/'grasp_video_dev_v1/grasp'/f'{EP}__f000'/'masks.npz'
    scene=Scene(str(world));scene.reset();apply_state(scene,json.loads(state.read_text())['objects'],require_matched=True,robot=json.loads(robot.read_text())['robot'])
    m=scene.env.sim.model;root=m.body_name2id('robot0_link6');distal=[]
    for g in range(m.ngeom):
        body=int(m.geom_bodyid[g]);yes=False
        while body:
            if body==root:yes=True;break
            body=int(m.body_parentid[body])
        distal.append(yes)
    raw=scene._seg_raw();rendered=(raw>=0)&np.asarray(distal)[np.clip(raw,0,None)];scene.env.close()
    with np.load(source) as a:anchor=a['robot']
    hand=anchor&binary_dilation(rendered,iterations=5);arm=anchor&~hand
    if min(hand.sum(),arm.sum())<30:raise ValueError('insufficient first-frame part support')
    dest=OUT/'anchors'/EP;dest.mkdir(parents=True);np.savez_compressed(dest/'masks.npz',robot_distal=hand,robot_proximal=arm)
    (dest/'association.json').write_text(json.dumps({'episode':EP,'objects':{n:{'status':'matched','identity_source':'shared Panda link6 subtree partitions frozen initial RGB robot anchor'} for n in ['robot_distal','robot_proximal']},'masks_sha256':sha(dest/'masks.npz')},indent=1))
    records=[];files={str(p):sha(p) for p in [source,state,robot]}
    for group,plan,evidence in SETS:
        for key in json.loads((R/plan/'selection.json').read_text())['selected']:
            plan_path=R/plan/key/'probe.json';probe=json.loads(plan_path.read_text());public=Path('/data_cf/cowmpo/evidence')/evidence/key;query=json.loads((public/'query.json').read_text())
            if query['query_sha256']!=sha(plan_path) or query['frames']!=probe['frames']:raise ValueError('action evidence mismatch')
            rec={'group':group,'key':key,'plan':str(plan_path),'public':str(public),'steps':probe['frames']};records.append(rec)
            for q in [plan_path,*[public/f'{EP}__f{s:03d}'/'frame0.png' for s in probe['frames']]]:files[str(q)]=sha(q)
            ready=OUT/'ready'/key
            for sub in ['initial','probes']:(ready/sub/EP).mkdir(parents=True)
            shutil.copy2(public/f'{EP}__f000/frame0.png',ready/'initial'/EP/'frame0.png');shutil.copy2(plan_path,ready/'probes'/EP/'probe.json')
    (OUT/'freeze.json').write_text(json.dumps({'queries':records,'files':files,'world_tree_sha256':tree_sha(world),'script_sha256':sha(__file__),'tracker_sha256':sha(SC/'track_probe_video_device.py'),
        'anchor_pixels':{'proximal':int(arm.sum()),'distal':int(hand.sum())},'rule':'partition original anchor using estimated initial shared Panda link6 subtree dilated5; propagate two IDs, union matched parts; no kinematic predictions after frame0; all four dev queries; post hoc diagnosis-driven change','GT_inputs':False},indent=1))
    print('ROBOT_PARTS_FROZEN',int(arm.sum()),int(hand.sum()),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:return prepare()
    frozen=json.loads((OUT/'freeze.json').read_text())
    if frozen['script_sha256']!=sha(__file__) or frozen['tracker_sha256']!=sha(SC/'track_probe_video_device.py'):raise ValueError('tracking code changed')
    def run(rec):
        key=rec['key'];ready=OUT/'ready'/key;parts=OUT/'parts'/key;final=OUT/'frames'/key;final.mkdir(parents=True)
        cmd=[sys.executable,str(SC/'track_probe_video_device.py'),'--data-role','dev','--device','cpu','--cpu-threads','8','--episodes',EP,'--anchors',str(OUT/'anchors'),'--initial-evidence',str(ready/'initial'),'--evidence',rec['public'],'--probes',str(ready/'probes'),'--out',str(parts)]
        with (OUT/(key+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((key,r.returncode))
        expected={f'{EP}__f{s:03d}' for s in rec['steps']}
        if {q.parent.name for q in parts.glob('*/association.json')}!=expected:raise ValueError('incomplete parts frame set')
        for frame in sorted(expected):
            source=parts/frame;meta=json.loads((source/'association.json').read_text());merged=np.zeros((256,256),bool)
            if sha(source/'masks.npz')!=meta['masks_sha256']:raise ValueError('parts mask hash mismatch')
            with np.load(source/'masks.npz') as a:
                for name,record in meta['objects'].items():
                    if record['status']=='matched':merged|=a[name]
            dest=final/frame;dest.mkdir();np.savez_compressed(dest/'masks.npz',robot=merged)
            record={'episode':frame,'algorithm':'shared_robot_parts_v4','objects':{'robot':{'status':'matched' if merged.any() else 'missing','identity_source':'union of independent proximal/distal first-frame-anchored trackers','parts':meta['objects']}},'masks_sha256':sha(dest/'masks.npz'),'source_sha256':sha(source/'association.json')}
            (dest/'association.json').write_text(json.dumps(record,indent=1))
        print('ROBOT_PARTS_QUERY_DONE',key,len(expected),flush=True);return {'key':key,'frames':len(expected)}
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,frozen['queries']))
    for p,h in frozen['files'].items():
        if sha(p)!=h:raise ValueError('evidence changed')
    (OUT/'complete.json').write_text(json.dumps(rows,indent=1));print('ROBOT_PARTS_DONE',sum(r['frames'] for r in rows),flush=True)
if __name__=='__main__':main()
