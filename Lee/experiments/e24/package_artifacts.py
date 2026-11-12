"""Archive fitted models and optimization logs, excluding row-level data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
names=['e16_cooperative','e17_contractive','e18_innovations','e19_temperature',
    'e20_coupled','e22_latent','e23_full_day','e24_fixed_initialization','e25_anchor']
files={}
for name in names:
    for path in (a.results/name).iterdir():
        if path.is_file() and path.suffix in ['.pkl','.pt','.json','.log']:
            files[f'{name}/{path.name}']=path
assert files and not any(Path(name).suffix in ['.npz','.csv','.xlsx'] for name in files)
index={'source_data_included':False,'residual_banks_included':False,
    'new_policy_training':False,'historical_defect_warning':'e22/e23 neural weights retain the dead NO3 initialization; e24 is corrected',
    'residual_regeneration':'Use authorized DARROW training data and the versioned experiment scripts.',
    'files':{name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in sorted(files.items())}}
a.out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name,path in sorted(files.items()):archive.write(path,name)
    archive.writestr('artifact_manifest.json',json.dumps(index,indent=2))
with zipfile.ZipFile(a.out) as archive:
    assert archive.testzip() is None
    for name,metadata in index['files'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==metadata['sha256']
index['archive']={'name':a.out.name,'bytes':a.out.stat().st_size,'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()}
a.index.write_text(json.dumps(index,indent=2));print(json.dumps(index['archive']),flush=True)
