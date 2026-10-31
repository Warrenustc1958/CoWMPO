import json
from pathlib import Path
import numpy as np
R=Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020/rack_vertical_v11_tracks"); T=Path("/data_cf/cowmpo/target_eval/codex_s56_20261030_1020/rack_vertical_dev_v11"); EP="libero_goal_t0_i0"
tot=[0,0]
for key in json.load(open(R/"freeze.json"))["keys"]:
    v=[]
    for d in sorted((R/"frames"/key).iterdir()):
        gt=np.load(T/key/d.name/"gt_masks.npz")["robot"]; p=np.load(d/"masks.npz")["robot"]
        v.append(float((gt&p).sum()/max((gt|p).sum(),1)))
    tot[0]+=len(v); tot[1]+=sum(x>=.8 for x in v)
    print(key, "frames", len(v), "mean", round(np.mean(v),4), "min", round(min(v),4), ">=0.8", sum(x>=.8 for x in v))
print("TOTAL", tot[1], "/", tot[0])
