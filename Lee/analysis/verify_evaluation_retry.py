"""Compare all retained overlapping generated evaluations after interruption."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True)
p.add_argument('--retry',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
old=json.loads((a.original/'manifest.json').read_text());new=json.loads((a.retry/'manifest.json').read_text())
for key in ('jobs','source_hashes','kind','environment','freeze_sha256','test_evaluated'):
    assert old[key]==new[key],f'Manifest mismatch: {key}'
assert not old['test_evaluated']
count=0;maximum=0.
def compare(left,right,path):
    global count,maximum
    if isinstance(left,dict):
        assert isinstance(right,dict) and left.keys()==right.keys(),path
        for key in left:compare(left[key],right[key],f'{path}/{key}')
    elif isinstance(left,list):
        assert isinstance(right,list) and len(left)==len(right),path
        for index,(x,y) in enumerate(zip(left,right)):compare(x,y,f'{path}/{index}')
    elif isinstance(left,(int,float)) and not isinstance(left,bool):
        assert isinstance(right,(int,float)) and not isinstance(right,bool),path
        np.testing.assert_allclose(left,right,rtol=1e-10,atol=1e-10,err_msg=path)
        count+=1;maximum=max(maximum,abs(float(left)-float(right)))
    else:assert left==right,path
files=[]
for path in sorted(a.original.glob('seed_*/*.json')):
    relative=path.relative_to(a.original);target=a.retry/relative
    assert target.exists(),f'Retry has not produced required overlap: {relative}'
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    compare(json.loads(path.read_text()),json.loads(target.read_text()),relative.as_posix())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    files.append({'path':relative.as_posix(),'original_sha256':before,
        'retry_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
assert files and {path['path'].split('/')[0] for path in files}=={f'seed_{i}' for i in range(5)}
result={'all_overlapping_generated_files_pass':True,'matching_manifests_identical':True,
    'original_files_unchanged':True,'files':files,'numeric_values_compared':count,
    'maximum_absolute_numeric_difference':maximum,'policy_retraining':False,
    'test_evaluated':False,'scope':'Interruption recovery equivalence only, not a policy improvement or full-result completion claim.'}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
