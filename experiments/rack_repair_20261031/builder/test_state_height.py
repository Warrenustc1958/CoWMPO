# Copyright 2026 The RLinf Authors.
"""Verify that an installed state preserves estimated height for free and fixed objects."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
root=Path(__file__).parent
import cowm
if Path(cowm.__file__).resolve()!=root/'toolkits/cowm/__init__.py':raise RuntimeError('wrong package path')
scene=Scene(str(R/'stove_parts_geometry_dev_v1/world07'));scene.reset()
state=json.loads((R/'stove_parts_states_dev_v1/libero_goal_t0_i0/state.json').read_text())['objects'];changed=copy.deepcopy(state)
changed['cream_cheese']['pos'][2]+=.05;changed['rack']['pos'][2]+=.02
apply_state(scene,changed,require_matched=True)
for name in ['cream_cheese','rack']:
    np.testing.assert_allclose(scene.env.object_pos(name),changed[name]['pos'],atol=1e-12)
raised_rgb=scene.rgb();apply_state(scene,state,require_matched=True)
for name in scene.names:np.testing.assert_allclose(scene.env.object_pos(name),state[name]['pos'],atol=1e-12)
if np.array_equal(raised_rgb,scene.rgb()):raise AssertionError('height installation must change rendered RGB')
scene.env.close();print('STATE_HEIGHT_CONTRACT_PASS free/fixed xyz and RGB',flush=True)
