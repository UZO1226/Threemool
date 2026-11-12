"""Package named follow-up runs without raw observations or residual banks."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
p.add_argument('--experiments',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--index',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
files={}
for name in a.experiments:
    assert Path(name).name==name
    folder=a.results/name;assert (folder/'summary.json').exists()
    for path in folder.iterdir():
        if path.is_file() and path.suffix in ['.pt','.pkl','.json','.log']:
            files[f'{name}/{path.name}']=path
manifest={'source_data_included':False,'residual_banks_included':False,'new_policy_training':False,
    'experiments':a.experiments,'files':{name:{'bytes':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in sorted(files.items())}}
a.out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name,path in sorted(files.items()):archive.write(path,name)
    archive.writestr('artifact_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(a.out) as archive:
    assert archive.testzip() is None
    for name,metadata in manifest['files'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==metadata['sha256']
manifest['archive']={'name':a.out.name,'bytes':a.out.stat().st_size,'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()}
a.index.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest['archive']),flush=True)
