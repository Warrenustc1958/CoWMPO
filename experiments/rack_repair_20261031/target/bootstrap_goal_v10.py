from pathlib import Path
import json,hashlib,shutil,zipfile
base=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v9');dest=base.with_name('codex_s56_20261030_1020_v10');shutil.copytree(base,dest)
a=Path('/data_cf/cowmpo/staging/codex_s56_20261030_1020/goal_delta_v10.zip')
with zipfile.ZipFile(a) as z:z.extractall(dest)
(dest/'goal_delta_manifest.json').write_text(json.dumps({'parent':str(base),'delta_sha256':hashlib.sha256(a.read_bytes()).hexdigest()},indent=1));print(dest)
