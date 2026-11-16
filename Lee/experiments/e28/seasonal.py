"""Run the corrected original model on a season-spanning training-only split."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e23'))
from full_day import main

if __name__=='__main__':
    reference=Path(sys.argv[sys.argv.index('--reference-summary')+1])
    config=json.loads(reference.read_text());assert config['split_checks_passed']
    main(score_sd=config['scoring_sd'])
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result['inner_partition']='timestamp-spaced 30 of official training188'
    result['scoring_sd']=config['scoring_sd'];result['model_sd_source']='new158 fitting dates only'
    result['reference_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
