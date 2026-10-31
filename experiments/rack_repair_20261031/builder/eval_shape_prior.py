# Copyright 2026 The RLinf Authors.
"""Same frozen shape and evidence, prior dynamics baseline for the S6 ablation."""
import json
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cowm import roles
from cowm.world.artifact import tree_sha
from cowm.s6.s6_identify_v2 import g2_eval,objective
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');READY=R/'s6_ready_v1'
OUT=R/'s6_shape_prior_v1';OUT.mkdir(exist_ok=False)
frozen=R/'s6_identification_v1';world=R/'world_pair_v1';world_hash=tree_sha(world)
eps=sorted(p.name for p in (READY/'probes').iterdir() if p.is_dir())
roles.require(eps,{'dev','fit'},'S6 prior baseline')
record={'params':{},'world_tree_sha256':world_hash}
with tempfile.TemporaryDirectory(dir=OUT) as tmp,ThreadPoolExecutor(12) as pool:
    for role in ['fit','dev']:
        summary,rows=g2_eval(pool,str(world),str(READY/'shape_states'),str(READY/'probes'),str(frozen/'probe_masks'),
            roles.select(eps,role),{},tmp,world_hash,str(frozen/'validity.json'),chunks=12)
        record[role]={'summary':summary,'J':objective(summary),'episodes':rows}
        print('SHAPE_PRIOR',role,json.dumps(summary),flush=True)
(OUT/'report.json').write_text(json.dumps(record,indent=1))
