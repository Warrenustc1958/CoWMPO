import hashlib
import json
import shutil
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
repo = Path('/mnt/nfs/fjj/Code/CoWMPO')
p6 = Path('/data_cf/cowmpo/staging/p6_v1')
unpacked = root / 'payload'
with zipfile.ZipFile(root / 'cowmpo_local_patch_20261030.zip') as z:
    z.extractall(unpacked)
records = json.loads((unpacked / 'patch_sources.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
verified = []
for r in records:
    if r['baseline_sha256'] is None:
        verified.append({**r, 'verified_remote_source': None})
        continue
    paths = [repo / r['path'], p6 / r['path'], p6 / Path(r['path']).name]
    matches = [p for p in paths if p.is_file() and (sha(p) == r['baseline_sha256'] or
               hashlib.sha256(p.read_text().encode()).hexdigest() == r.get('baseline_lf_sha256'))]
    if not matches:
        raise RuntimeError('baseline mismatch: ' + r['path'] + ' ' + str({str(p):sha(p) for p in paths if p.is_file()}))
    verified.append({**r, 'verified_remote_source': str(matches[0]), 'remote_sha256': sha(matches[0])})
stage = root / 'repo'
stage.mkdir()
shutil.copytree(repo / 'toolkits', stage / 'toolkits', ignore=shutil.ignore_patterns('__pycache__'))
(stage / 'tests').mkdir()
shutil.copytree(repo / 'tests/cowm', stage / 'tests/cowm')
for relative in ['toolkits/cowm/init/associate.py', 'toolkits/cowm/init/test_associate.py',
                 'toolkits/cowm/init/test_recheck.py', 'toolkits/cowm/init/perceive.py']:
    shutil.copyfile(p6 / relative, stage / relative)
for r in verified:
    dst = stage / r['path']
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(unpacked / r['path'], dst)
(root / 'verified_remote_sources.json').write_text(json.dumps(verified,indent=2))
print('BASELINES_VERIFIED',len(verified),'STAGE',stage)
