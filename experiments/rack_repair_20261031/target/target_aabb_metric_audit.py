"""Compare conservative per-geom bounds and tight transformed mesh bounds; no geometry export."""
import json
from pathlib import Path
import numpy as np
T=Path('/data_cf/cowmpo/target_eval/codex_s56_20261030_1020')
d=json.loads((T/'aabb_diagnostic.json').read_text());report={}
for n,r in d.items():
    report[n]={'conservative_vs_tight_center_xy_cm':float(np.linalg.norm((np.asarray(r['old_center'])-r['mesh_exact_center'])[:2])*100),'conservative_vs_tight_extent_max_cm':float(np.max(np.abs(np.asarray(r['old_half'])-r['mesh_exact_half']))*200)}
(T/'aabb_metric_audit_v1.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
