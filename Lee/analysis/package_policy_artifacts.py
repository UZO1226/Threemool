"""Archive completed policy runs, excluding raw inputs and stochastic banks."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--freeze',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
files={};included=[];results=a.results.resolve()
for folder in a.runs:
    folder=folder.resolve();relative=folder.relative_to(results)
    metrics=json.loads((folder/'metrics.json').read_text());progress=json.loads((folder/'progress.json').read_text())
    config=json.loads((folder/'effective_config.json').read_text())
    assert progress['phase']=='complete' and metrics['steps']==401280
    assert not config['experiment_metadata']['test_evaluated']
    included.append({'run':relative.as_posix(),'method':metrics['method'],'policy_seed':metrics['seed'],
        'environment':config['experiment_metadata']['environment'],'training_steps':metrics['steps']})
    for path in folder.rglob('*'):
        if path.is_file() and path.suffix in ('.pt','.json'):
            files[(relative/path.relative_to(folder)).as_posix()]=path
    log=folder.parent.parent/f'{folder.parent.name}_s{metrics["seed"]}.log'
    if log.exists():files[f'{relative.parts[0]}/{log.name}']=log
    source_manifest=folder.parent.parent/'manifest.json'
    if source_manifest.exists():files[f'{relative.parts[0]}/source_run_manifest.json']=source_manifest
files['metadata/environment_freeze.json']=a.freeze
license_path=Path(__file__).resolve().parents[1]/'code/darrow/FCSRL_LICENSE'
files['metadata/FCSRL_LICENSE']=license_path
manifest={'new_policy_training':True,'source_data_included':False,'residual_banks_included':False,
    'pit_banks_included':False,'included_completed_runs':included,
    'scope':'Only listed completed policy subsets. No improvement claim from archive existence.',
    'files':{name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for name,path in sorted(files.items())}}
a.out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name,path in sorted(files.items()):archive.write(path,name)
    archive.writestr('artifact_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(a.out) as archive:
    assert archive.testzip() is None
    for name,metadata in manifest['files'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==metadata['sha256']
manifest['archive']={'name':a.out.name,'bytes':a.out.stat().st_size,'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()}
a.index.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest['archive']),flush=True)
