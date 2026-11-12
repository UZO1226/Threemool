"""Package model artifacts and logs separately from Git, excluding source data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser()
p.add_argument('--runs',type=Path,required=True)
p.add_argument('--audit',type=Path,required=True)
p.add_argument('--old-model',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--index',type=Path,required=True)
a=p.parse_args()
if a.out.exists():
    raise FileExistsError(a.out)
files={}
for path in a.audit.iterdir():
    if path.name in ['direct_7.pkl','direct_15.pkl','persistence.pkl'] or path.suffix == '.json':
        files[f'environment/{path.name}']=path
files['environment/old_delta_gbm.pkl']=a.old_model
for path in a.runs.rglob('*'):
    if not path.is_file(): continue
    relative=path.relative_to(a.runs)
    if path.suffix in ['.json','.log'] and 'model' not in relative.parts and 'model_final' not in relative.parts:
        files[f'agent_runs/{relative.as_posix()}']=path
for metrics in a.runs.glob('*/seed_*/metrics.json'):
    run=metrics.parent
    best=json.loads(metrics.read_text())['best_epoch']
    for directory in [run/'model_final',run/'model'/f'epoch_{best}']:
        for path in directory.rglob('*'):
            if path.is_file():
                files[f'agent_runs/{path.relative_to(a.runs).as_posix()}']=path
license_path=Path(__file__).resolve().parents[2]/'code/darrow/FCSRL_LICENSE'
files['licenses/FCSRL_LICENSE']=license_path
index={'source_data_included':False,'all_final_and_selected_checkpoints':True,
       'files':{name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
                for name,path in sorted(files.items())}}
a.out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name,path in sorted(files.items()): archive.write(path,name)
    archive.writestr('artifact_manifest.json',json.dumps(index,indent=2))
index['archive']={'name':a.out.name,'bytes':a.out.stat().st_size,'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()}
a.index.write_text(json.dumps(index,indent=2))
print(json.dumps(index['archive']),flush=True)
