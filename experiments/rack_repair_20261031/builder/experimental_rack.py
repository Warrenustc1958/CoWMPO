# Copyright 2026 The RLinf Authors.
"""Generic slatted two-shelf fixture, built solely from dimensions and an incline parameter."""
import math
from cowm.world.templates import _part, _save, _wrap


def build(name,out_dir,width,depth,height,incline,density=600.,friction=.7):
    rgba='0.45 0.3 0.2 1'
    def part(kind,size,pos,euler=None):
        return _part(kind,size,pos,rgba,euler=euler,density=density,friction=friction)
    thickness=.008
    slope=math.sin(incline);co=math.cos(incline)
    upper=height-.45*depth*abs(slope)-thickness
    lower=max(.05,.45*upper)
    body=part('box',[depth/2,width/2,.004],[0,0,.004])
    for level in [lower,upper]:
        for j in range(8):
            x=(-.45+.9*j/7)*depth
            body+=part('box',[.035*depth,width/2,thickness/2],
                       [co*x,0,level-slope*x],euler=f'0 {incline} 0')
        body+=part('box',[.012,width/2,.006],[0,0,level-.012])
    for side in [-1,1]:
        body+=part('box',[.008,.008,height/2],[.25*depth,side*.35*width,height/2])
    return _save(out_dir,name,_wrap(name,'',body,height,max(width,depth)/2))
