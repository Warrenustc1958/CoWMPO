"""Compare original and generic rack geometries with the same four-yaw-start estimator, dev only."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cowm.init import estimate_state as ES
from fit_rack_dev import worker

R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--world');parser.add_argument('--out');a=parser.parse_args()
    if a.world:
        original=ES.estimate
        def multi(scene,masks,association=None,allow_legacy=False):
            rack=next(o for o in scene.world['objects'] if o['name']=='rack')
            nominal=rack.get('yaw',0.);best=None
            for yaw in [0.,math.pi/2,math.pi,-math.pi/2]:
                scene.reset();rack['yaw']=yaw
                result=original(scene,masks,association,allow_legacy)
                if best is None or result[1]>best[1]:best=result
            rack['yaw']=nominal
            for index,obj in enumerate(scene.world['objects']):
                state=best[0][obj['name']];scene.set_pose(index,state['pos'][:2],state['yaw'])
            return best
        ES.estimate=multi
        return worker(a.world,a.out)
    out=R/'rack_multistart_dev_v1';out.mkdir(exist_ok=False)
    worlds=[Path('/data_cf/cowmpo/worlds/s5_v4/shape/world_fit')]+sorted((R/'rack_geometry_dev_v1').glob('world*'))
    def run(item):
        index,world=item;state=out/f'state{index:02d}'
        with (out/f'job{index:02d}.log').open('w') as log:
            p=subprocess.run([sys.executable,__file__,'--world',str(world),'--out',str(state)],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError((index,p.returncode))
        score=json.loads((state/'score.json').read_text())
        print('multistart_candidate',index,score['rack'],flush=True)
        return {'index':index,'world':str(world),'state':str(state),'score':score}
    with ThreadPoolExecutor(6) as pool:records=list(pool.map(run,enumerate(worlds)))
    best=max(records,key=lambda r:r['score']['rack'])
    (out/'selection.json').write_text(json.dumps({'candidates':records,'best':best,'selection':'dev SAM IoU only'},indent=2))
    print('MULTISTART_DONE',json.dumps(best))

if __name__=='__main__':main()
