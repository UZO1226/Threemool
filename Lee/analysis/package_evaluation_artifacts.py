"""Preserve full generated matched-air evaluations without source plant data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/e44'))
from report import summarize

p=argparse.ArgumentParser()
p.add_argument('--pairs',type=Path,nargs='+',required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--index',type=Path,required=True)
p.add_argument('--interrupted-roots',type=Path,nargs='*',default=[])
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
pairs=[json.loads(path.read_text()) for path in a.pairs]
report=summarize(pairs)
files={}
for path,pair in zip(a.pairs,pairs):
    assert path.name=='summary.json'
    seed=pair['policy_seed']
    for item in path.parent.iterdir():
        if item.is_file():
            assert item.suffix=='.json',f'Unexpected generated evaluation file: {item}'
            files[f'seed_{seed}/{item.name}']=item
    log=path.parent.parent/f'seed_{seed}.log'
    if log.exists():files[f'seed_{seed}/runner.log']=log
    manifest=path.parent.parent/'manifest.json'
    if manifest.exists():files[f'seed_{seed}/run_manifest.json']=manifest
interrupted=[]
for number,root in enumerate(a.interrupted_roots):
    assert not list(root.glob('seed_*/summary.json')),'Use completed seed pairs for finished evaluations'
    retained=[]
    for item in sorted(root.rglob('*')):
        if item.is_file():
            assert item.suffix in ('.json','.log'),f'Unexpected partial evaluation file: {item}'
            name=f'interrupted_{number}/{item.relative_to(root).as_posix()}'
            files[name]=item;retained.append(name)
    assert retained
    interrupted.append({'root':str(root),'files':retained,
        'scope':'Retained incomplete outputs. Their old running status is not a liveness claim.'})
metadata={'source_data_included':False,'residual_banks_included':False,
    'pit_banks_included':False,'new_policy_training':False,
    'scope':'Generated frozen-surrogate evaluation rows and all air-only queries. Not source plant data.',
    'policy_seeds':report['policy_seeds'],'candidate_method':report['candidate_method'],
    'meets_prespecified_simulation_criterion':report['meets_prespecified_simulation_criterion'],
    'interrupted_outputs':interrupted,
    'files':{name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for name,path in sorted(files.items())}}
a.out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name,path in sorted(files.items()):archive.write(path,name)
    archive.writestr('five_seed_report.json',json.dumps(report,indent=2,allow_nan=False))
    archive.writestr('artifact_manifest.json',json.dumps(metadata,indent=2))
with zipfile.ZipFile(a.out) as archive:
    assert archive.testzip() is None
    for name,entry in metadata['files'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==entry['sha256']
metadata['archive']={'name':a.out.name,'bytes':a.out.stat().st_size,
    'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()}
a.index.write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata['archive']),flush=True)
