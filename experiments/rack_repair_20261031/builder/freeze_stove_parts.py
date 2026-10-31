# Copyright 2026 The RLinf Authors.
"""Freeze dev-selected geometry and completion before aggregate fit validation."""
import hashlib
import json
from pathlib import Path
from cowm.world.artifact import tree_sha
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020');SC=Path(__file__).parent
world=R/'stove_parts_geometry_dev_v1/world07'
files=[SC/'complete_functional_masks.py',SC/'refit_frozen_geometry.py',SC/'fit_stove_parts_dev.py',
       R/'stove_parts_geometry_dev_v1/selection.json',R/'visual_dev_analysis_v1/report.npz']
record={'world':str(world),'world_tree_sha256':tree_sha(world),
 'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
 'selection_scope':'six dev RGB/masks; 32 dev aggregate validation; no fit geometry results yet',
 'completion':{'scope':'objects with program-declared joints','min_proposal_score':.2,'area_ratio':[1.05,1.6],
  'min_anchor_containment':.95,'max_other_or_robot_fraction':.1,'reject_visual_negative':True},
 'gate_test_queried':False}
with (R/'stove_parts_freeze_v1.json').open('x') as f:json.dump(record,f,indent=1)
print('STOVE_PARTS_FROZEN',record['world_tree_sha256'])
