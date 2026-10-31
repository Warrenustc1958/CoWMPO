# Copyright 2026 The RLinf Authors.
"""Protocol boundary checks for release labels and acquisition coverage."""
import json
import numpy as np
from plan_release_dev_v2 import classify_release

def run():
    initial=np.zeros(3);held=np.array([0,0,.1]);stationary=np.zeros((100,3))
    cases=[
        ('stable_support',2,np.zeros(3),stationary,[False]*100,[True]*100,'stable'),
        ('drop',2,np.array([0,0,.14]),np.c_[np.zeros((100,2)),np.r_[np.linspace(.14,0,60),np.zeros(40)]],[False]*100,[True]*100,'dropped'),
        ('no_carry_is_not_stable',0,np.zeros(3),stationary,[False]*100,[True]*100,'no_carry'),
        ('not_detached',2,np.zeros(3),stationary,[True]*100,[True]*100,'unresolved'),
        ('no_support',2,np.zeros(3),stationary,[False]*100,[False]*100,'unresolved'),
        ('midrange_fall',2,np.array([0,0,.04]),stationary,[False]*100,[True]*100,'unresolved'),
        ('continued_motion',2,np.zeros(3),np.c_[np.linspace(0,.04,100),np.zeros((100,2))],[False]*100,[True]*100,'unresolved'),
    ]
    for name,fingers,release,trajectory,robot,support,expected in cases:
        got=classify_release(initial,held,fingers,release,trajectory,robot,support)['status']
        if got!=expected:raise AssertionError((name,expected,got))
    print(json.dumps({'passed':len(cases),'cases':[c[0] for c in cases]}))
if __name__=='__main__':run()
