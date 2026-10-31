# Copyright 2026 The RLinf Authors.
"""Private audit of tracking bias in frozen rack responses; never candidate-selection input."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from cowm.init.estimate_state import Scene,iou
from cowm.s6.plan_probe import apply_state
from cowm.world.artifact import verify_tree
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020');OUT=T/'rack_response_mask_audit_v1';CFG=R/'rack_isolated_candidates_dev_v3/freeze.json';EP='libero_goal_t0_i0'

def worker(name):
    cfg=json.loads(CFG.read_text());world,states=map(Path,cfg['candidates'][name]);verify_tree(world,cfg['worlds'][str(world)])
    scene=Scene(str(world));robot=json.loads((R/'robot_rgb_all_dev_v2'/EP/'robot.json').read_text())['robot'];rows=[]
    for rec in cfg['queries']:
        scene.reset();apply_state(scene,json.loads((states/EP/'state.json').read_text())['objects'],require_matched=True,robot=robot)
        probe=json.loads(Path(rec['source']).read_text());values=[];public_values=[]
        private=T/('rack_isolated_dev_v2' if rec['group']=='horizontal' else 'rack_vertical_dev_v3')/rec['key']
        for step,act in enumerate(probe['actions'],1):
            scene.env.step(np.asarray(act))
            if step not in rec['frames']:continue
            frame=f'{EP}__f{step:03d}'
            with np.load(private/frame/'gt_masks.npz') as a:target=a['robot']
            with np.load(Path(rec['video'])/frame/'masks.npz') as a:tracked=a['robot']
            rendered=scene.robot_mask();values.append(iou(rendered,target));public_values.append(iou(rendered,tracked))
            if rec['key']=='vert_x+000_y+000' and step==340:Image.fromarray(scene.rgb()).save(OUT/(name+'_center340.png'))
        rows.append({'key':rec['key'],'gt_iou':float(np.mean(values)),'sam_iou':float(np.mean(public_values))})
    scene.env.close();(OUT/(name+'.json')).write_text(json.dumps({'name':name,'gt_iou':float(np.mean([r['gt_iou'] for r in rows])),'sam_iou':float(np.mean([r['sam_iou'] for r in rows])),'queries':rows},indent=1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate');a=p.parse_args()
    if a.candidate:return worker(a.candidate)
    OUT.mkdir(exist_ok=False);cfg=json.loads(CFG.read_text())
    def run(name):
        with (OUT/(name+'.log')).open('w') as log:r=subprocess.run([sys.executable,__file__,'--candidate',name],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError((name,r.returncode))
        return json.loads((OUT/(name+'.json')).read_text())
    with ThreadPoolExecutor(3) as pool:rows=list(pool.map(run,cfg['candidates']))
    (OUT/'summary.json').write_text(json.dumps({'scope':'Target-private mask-quality attribution only; GT scores cannot select or tune Builder worlds','models':rows},indent=1))
    public=Path('/data_cf/cowmpo/evidence/codex_rack_vertical_dev_v3/vert_x+000_y+000')/f'{EP}__f340/frame0.png'
    canvas=Image.new('RGB',(1024,280),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,path) in enumerate([('Target public RGB',public)]+[(n,OUT/(n+'_center340.png')) for n in cfg['candidates']]):
        draw.text((i*256+4,4),label,fill='black');canvas.paste(Image.open(path),(i*256,24))
    canvas.save(OUT/'center340_comparison.png');print(json.dumps(rows,indent=1),flush=True)
if __name__=='__main__':main()
