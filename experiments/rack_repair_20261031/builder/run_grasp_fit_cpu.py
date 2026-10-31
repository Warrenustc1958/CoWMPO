# Copyright 2026 The RLinf Authors.
"""Run verified fit-only SAM2 video inference on CPU without sharing Ray GPUs."""
import hashlib,json,shutil,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from cowm import roles
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SC=Path(__file__).parent;SRC=R/'grasp_tracking_ready_fit_v2/grasp';OUT=R/'grasp_video_fit_cpu_v2';OUT.mkdir(exist_ok=False);EPS=sorted(p.name for p in (SRC/'anchors').iterdir());roles.require(EPS,{'fit'},'CPU grasp video');pilot=json.loads((R/'cpu_gpu_tracker_pilot_comparison.json').read_text())
if not pilot['identical_statuses'] or pilot['min_iou']<.98:raise ValueError('CPU pilot not equivalent')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads((R/'video_freeze_v1.json').read_text())
for p,h in frozen['files'].items():
    if sha(Path(p))!=h:raise ValueError('frozen model or perception source changed')
expected={f'{ep}__f{step:03d}' for ep in EPS for step in json.loads((SRC/'probes'/ep/'probe.json').read_text())['frames']};config={'episodes':EPS,'role':'fit','device':'CPU','processes':4,'threads_per_process':8,'script_sha256':sha(Path(__file__)),'tracker_sha256':sha(SC/'track_probe_video_device.py'),'ready_freeze_sha256':sha(R/'grasp_tracking_ready_fit_v2/freeze.json'),'model_freeze_sha256':sha(R/'video_freeze_v1.json'),'pilot_comparison':pilot,'GT_inputs':False};(OUT/'freeze.json').write_text(json.dumps(config,indent=1))
def run(i):
    args=[sys.executable,str(SC/'track_probe_video_device.py'),'--data-role','fit','--device','cpu','--cpu-threads','8','--episodes',','.join(EPS[i::4]),'--out',str(OUT/f'part{i}')]
    for key,src in [('anchors','anchors'),('initial-evidence','initial'),('evidence','evidence'),('probes','probes')]:args+=['--'+key,str(SRC/src)]
    with (OUT/f'part{i}.log').open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((i,r.returncode))
    print('CPU_GRASP_VIDEO_PART_DONE',i,flush=True)
with ThreadPoolExecutor(4) as pool:list(pool.map(run,range(4)))
frames=OUT/'frames';frames.mkdir();actual=set()
for i in range(4):
    for p in sorted((OUT/f'part{i}').glob('*/association.json')):
        if p.parent.name in actual:raise ValueError('duplicate video frame')
        actual.add(p.parent.name);shutil.copytree(p.parent,frames/p.parent.name)
if actual!=expected:raise ValueError('incomplete exact frame set')
(OUT/'complete.json').write_text(json.dumps({'episodes':28,'frames':len(actual),'parts':[json.loads((OUT/f'part{i}/identity_summary.json').read_text()) for i in range(4)],'device':'CPU','GPU_processes_started':0},indent=1));print('CPU_GRASP_VIDEO_COMPLETE',28,len(actual),flush=True)
