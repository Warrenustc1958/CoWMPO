import json
from pathlib import Path
R=Path('/data_cf/cowmpo/worlds/codex_s56_20261030_1020')
w=json.loads((R/'stove_parts_geometry_dev_v1/world07/world.json').read_text())
print('BUILDER_CHEESE',next(o for o in w['objects'] if o['name']=='cream_cheese'))
g=json.loads(Path('/data_cf/cowmpo/target_eval/s5_gt/dev3/libero_goal_t0_i0.json').read_text())
print('GT_KEYS_ONLY',list(g['cream_cheese']))
