from pathlib import Path
import hashlib,json,shutil,zipfile
base=Path('/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v6');dest=base.with_name('codex_s56_20261030_1020_v7')
shutil.copytree(base,dest)
archive=Path('/data_cf/cowmpo/staging/codex_s56_20261030_1020/robot_delta_v7.zip')
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if not (dest/name).resolve().is_relative_to(dest):raise ValueError('invalid archive path')
    z.extractall(dest)
(dest/'robot_delta_manifest.json').write_text(json.dumps({'parent':str(base),'delta_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=1))
print('ROBOT_STAGING_CREATED',dest)
