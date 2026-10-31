from pathlib import Path
import hashlib,json,shutil,zipfile
base=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v8');dest=base.with_name('codex_s56_20261030_1020_v9');shutil.copytree(base,dest)
a=Path('/data_cf/cowmpo/staging/codex_s56_20261030_1020/backend_delta_v9.zip')
with zipfile.ZipFile(a) as z:z.extractall(dest)
(dest/'backend_delta_manifest.json').write_text(json.dumps({'parent':str(base),'delta_sha256':hashlib.sha256(a.read_bytes()).hexdigest()},indent=1));print(dest)
