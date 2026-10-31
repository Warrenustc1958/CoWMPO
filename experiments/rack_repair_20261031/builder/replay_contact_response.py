# Copyright 2026 The RLinf Authors.
"""Record code-world response to frozen action queries, without Target internals."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');OUT=R/'contact_response_dev_v1'
def worker(key):
    rack=key.startswith('h')
    path=R/'rack_touch_dev_v1'/key/'probes/libero_goal_t0_i0/probe.json' if rack else R/'grasp_contrast_dev_v1'/key/'probe.json'
    p=json.loads(path.read_text());ep=p['episode'];dest=OUT/key;dest.mkdir()
    scene=Scene(str(R/'stove_parts_geometry_dev_v1/world07'));scene.reset();apply_state(scene,json.loads((R/'stove_parts_states_dev_v1'/ep/'state.json').read_text())['objects'],require_matched=True)
    eef=[]
    for step,act in enumerate(p['actions'],1):
        scene.env.step(np.asarray(act));eef.append(scene.env.eef_pos().tolist())
        if step in p['frames']:
            frame=dest/f'f{step:03d}';frame.mkdir();seg=scene.seg()
            np.savez_compressed(frame/'masks.npz',robot=scene.robot_mask(),cream_cheese=seg==scene.names.index('cream_cheese'))
    scene.env.close();np.save(dest/'eef.npy',eef);print('replay_complete',key,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--key');a=p.parse_args()
    if a.key:return worker(a.key)
    OUT.mkdir(exist_ok=False);keys=[f'libero_goal_t{k}_i0__grasp' for k in range(6)]+['h070','h170','h270']
    def run(key):
        with (OUT/(key+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--key',key],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((key,r.returncode))
    with ThreadPoolExecutor(6) as pool:list(pool.map(run,keys))
    print('CONTACT_REPLAY_DONE',flush=True)
if __name__=='__main__':main()
