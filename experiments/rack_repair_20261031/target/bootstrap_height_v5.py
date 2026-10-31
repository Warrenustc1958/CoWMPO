from pathlib import Path
import hashlib
import json
import shutil
import zipfile
base=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v4')
dest=base.with_name('codex_s56_20261030_1020_v5')
shutil.copytree(base,dest)
archive=Path('/data_cf/cowmpo/staging/codex_s56_20261030_1020/height_delta_v5.zip')
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        target=(dest/name).resolve()
        if not target.is_relative_to(dest):raise ValueError('invalid archive path')
    z.extractall(dest)
(dest/'height_delta_manifest.json').write_text(json.dumps({'parent':str(base),'delta_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=1))
print('HEIGHT_STAGING_CREATED',dest)
