# Copyright 2026 The RLinf Authors.
"""Generic crossed slatted rack with an explicit common support plane."""
import math
import numpy as np
from cowm.world.templates import _part,_save,_wrap

def parts(width,depth,height):
    angle=math.atan2(height,depth);length=math.hypot(height,depth);rows=[]
    for sign in [-1,1]:
        tilt=sign*angle
        for j in range(12):
            t=(-.5+(j+.5)/12)*length
            rows.append((np.array([length/12*.43,width/2,.003]),np.array([t*math.cos(tilt),0,height/2-t*math.sin(tilt)]),tilt))
        for side in [-1,1]:
            rows.append((np.array([length/2,.007,.007]),np.array([0,side*(width/2+.005),height/2]),tilt))
    bottom=min(p[2]-abs(math.sin(t))*h[0]-abs(math.cos(t))*h[2] for h,p,t in rows)
    return [(h,p-np.array([0,0,bottom]),t) for h,p,t in rows]

def build(name,out_dir,width,depth,height):
    rows=parts(width,depth,height);body=''
    for half,pos,tilt in rows:
        rail=half[1]<.01
        body+=_part('box',half.tolist(),pos.tolist(),'0.3 0.15 0.08 1' if rail else '0.55 0.4 0.25 1',euler=f'0 {tilt} 0',density=600.,friction=.7)
    top=max(p[2]+abs(math.sin(t))*h[0]+abs(math.cos(t))*h[2] for h,p,t in rows)
    return _save(out_dir,name,_wrap(name,'',body,top,max(width,depth)/2)),top
