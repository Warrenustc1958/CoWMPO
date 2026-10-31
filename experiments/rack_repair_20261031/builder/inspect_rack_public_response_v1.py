# Copyright 2026 The RLinf Authors.
"""Compare public rack response images with frozen Builder predictions."""
from pathlib import Path
from PIL import Image,ImageDraw
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');E=Path('/data_cf/cowmpo/evidence/codex_rack_multidir_dev_v1');OUT=R/'rack_public_response_review_v1';OUT.mkdir(exist_ok=False);EP='libero_goal_t0_i0'
for key in ['xp_h220','xm_h220','ym_h100','yp_h100']:
    canvas=Image.new('RGB',(3*256,2*280),(245,245,245));draw=ImageDraw.Draw(canvas)
    for col,step in enumerate([190,240,300]):
        frame=f'{EP}__f{step:03d}'
        for row,(label,path) in enumerate([('Target RGB',E/key/frame/'frame0.png'),('Builder',R/'rack_multidir_preflight_dev_v1'/key/'base'/frame/'frame0.png')]):
            canvas.paste(Image.open(path),(col*256,row*280+24));draw.text((col*256+4,row*280+5),f'{label} {key} step {step}',fill=(0,0,0))
    canvas.save(OUT/(key+'.png'))
print('PUBLIC_RESPONSE_REVIEW_COMPLETE')
