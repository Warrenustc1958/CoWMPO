# Copyright 2026 The RLinf Authors.
"""Isolate the SDK multi-contact option using the same frozen dev actions and RGB."""
import argparse, hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene,iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import tree_sha,verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'multiccd_grasp_dev_v1';WORLD=R/'bowl_support_dev_v1/world11';EPS=[f'libero_goal_t{k}_i0' for k in range(6)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worker(flag):
    scene=Scene(str(WORLD));rows=[];out=OUT/('on' if flag else 'off');out.mkdir()
    for ep in EPS:
        for mode in ['grasp','miss']:
            scene.reset();scene.env.sim.model.opt.enableflags=(int(scene.env.sim.model.opt.enableflags)|16) if flag else (int(scene.env.sim.model.opt.enableflags)&~16)
            apply_state(scene,json.loads((R/'natural_states_all_dev_v1'/ep/'state.json').read_text())['objects'],require_matched=True,robot=json.loads((R/'robot_rgb_all_dev_v2'/ep/'robot.json').read_text())['robot'])
            probe=json.loads((R/'grasp_contrast_dev_v1'/(ep+'__'+mode)/'probe.json').read_text());k=scene.names.index('cream_cheese');initial=scene.env.object_pos('cream_cheese').copy();fingers=set();scores=[];contacts=[]
            for step,act in enumerate(probe['actions'],1):
                scene.env.step(np.asarray(act))
                if mode=='grasp' and 190<=step<=240 and step%10==0:
                    path=R/'grasp_video_dev_v1/grasp'/f'{ep}__f{step:03d}';rec=json.loads((path/'association.json').read_text())
                    if rec['masks_sha256']!=sha(path/'masks.npz') or rec['objects']['cream_cheese']['status']!='matched':raise ValueError('invalid response mask')
                    with np.load(path/'masks.npz') as a:target=a['cream_cheese']
                    scores.append(iou(scene.seg()==k,target))
                if step==probe['phases']['holding']:
                    held=scene.env.object_pos('cream_cheese').copy();sim=scene.env.sim
                    for c in sim.data.contact[:sim.data.ncon]:
                        for g,r in [(int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))]:
                            if scene.geom2obj[g]==k and 'finger' in (sim.model.geom_id2name(r) or ''):fingers.add(int(sim.model.geom_bodyid[r]))
            result={'key':ep+'__'+mode,'episode':ep,'prediction':bool(held[2]-initial[2]>.05 and len(fingers)>=2),'response_iou':scores}
            (out/(result['key']+'.json')).write_text(json.dumps(result,indent=1));rows.append(result)
    scene.env.close();(out/'freeze.json').write_text(json.dumps({'keys':[r['key'] for r in rows]},indent=1))
    report={'multiccd':bool(flag),'mean_grasp_response_iou':float(np.mean([v for r in rows for v in r['response_iou']])),
        'grasp_predictions':[r['prediction'] for r in rows if r['key'].endswith('__grasp')],'miss_predictions':[r['prediction'] for r in rows if r['key'].endswith('__miss')]}
    (out/'summary.json').write_text(json.dumps(report,indent=1))
def main():
    p=argparse.ArgumentParser();p.add_argument('--flag',type=int);a=p.parse_args()
    if a.flag is not None:return worker(a.flag)
    OUT.mkdir(exist_ok=False);digest=tree_sha(WORLD);files={}
    for ep in EPS:
        for q in [R/'natural_states_all_dev_v1'/ep/'state.json',R/'robot_rgb_all_dev_v2'/ep/'robot.json',*[R/'grasp_contrast_dev_v1'/(ep+'__'+mode)/'probe.json' for mode in ['grasp','miss']]]:files[str(q)]=sha(q)
    (OUT/'freeze.json').write_text(json.dumps({'world_tree_sha256':digest,'files':files,'script_sha256':sha(__file__),'GT_inputs':False,'scope':'platform-option diagnostic, not a world repair or a proposal to disable multi-contact globally'},indent=1))
    def run(flag):
        with (OUT/f'flag{flag}.log').open('w') as log:r=subprocess.run([sys.executable,__file__,'--flag',str(flag)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((flag,r.returncode))
        return json.loads((OUT/('on' if flag else 'off')/'summary.json').read_text())
    with ThreadPoolExecutor(2) as pool:rows=list(pool.map(run,[0,1]))
    verify_tree(WORLD,digest)
    for q,h in files.items():
        if sha(q)!=h:raise ValueError('diagnostic input changed')
    (OUT/'report.json').write_text(json.dumps(rows,indent=1));print(json.dumps(rows,indent=1),flush=True)
if __name__=='__main__':main()
