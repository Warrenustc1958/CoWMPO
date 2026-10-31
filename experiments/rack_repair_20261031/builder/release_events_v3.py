# Copyright 2026 The RLinf Authors.
"""Measure release from sustained loss of robot contact, including premature slip."""
import numpy as np

def classify_event(initial,held,fingers,positions,robot_contact,support):
    points=np.asarray(positions,dtype=float);contact=np.asarray(robot_contact,dtype=bool)
    if points.shape!=(len(contact),3) or len(points)<21 or len(support)!=len(points):raise ValueError('incomplete release trace')
    acquired=bool(held[2]-initial[2]>.05 and fingers>=2)
    start=next((i for i in range(1,len(contact)-9) if not contact[i:i+10].any()),None)
    if not acquired:return {'status':'no_carry','acquired':False,'event_offset':start}
    if start is None:return {'status':'unresolved','acquired':True,'event_offset':None,'reason':'no sustained separation'}
    anchor=points[start-1];tail=points[start:]
    drop=float(max(0,anchor[2]-tail[:,2].min()));drift=float(np.linalg.norm(tail[:,:2]-anchor[:2],axis=1).max())
    span=float(np.linalg.norm(points[-20:]-points[-1],axis=1).max());detached=not contact[-20:].any();supported=sum(support[-20:])>=15
    status='dropped' if drop>.05 else ('stable' if drop<=.03 and drift<=.03 and span<=.002 and detached and supported else 'unresolved')
    return {'status':status,'acquired':True,'event_offset':start,'drop_m':drop,'horizontal_drift_m':drift,'tail_span_m':span,'detached':bool(detached),'supported':bool(supported)}

def self_test():
    initial=np.zeros(3);held=np.array([0,0,.14]);points=np.zeros((191,3));points[:41,2]=.14;points[41:61,2]=np.linspace(.14,0,20)
    out=classify_event(initial,held,2,points,[True]*41+[False]*150,[False]*61+[True]*130)
    assert out['status']=='dropped' and out['event_offset']==41
    # Deliberate lowering while still grasped is not a fall.
    points=np.zeros((191,3));points[:91,2]=np.linspace(.14,0,91)
    out=classify_event(initial,held,2,points,[True]*91+[False]*100,[False]*91+[True]*100)
    assert out['status']=='stable' and out['event_offset']==91
    assert classify_event(initial,held,0,points,[True]*91+[False]*100,[True]*191)['status']=='no_carry'
    assert classify_event(initial,held,2,points,[True]*191,[True]*191)['status']=='unresolved'
    return {'passed':4,'cases':['premature_slide','controlled_lowering','failed_acquisition','never_separated']}
if __name__=='__main__':print(self_test())
