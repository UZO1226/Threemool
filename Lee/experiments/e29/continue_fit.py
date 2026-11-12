"""Continue paired e27 weights with explicit optimizer reset and control."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e27'))
from slow_memory import SlowMemoryModel,cache,train_full_day

if __name__=='__main__':
    parent=ROOT.parent/'results/e27_slow_memory';source=json.loads((parent/'summary.json').read_text())
    for member in range(5):
        path=parent/f'epoch80_m{member}.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['checkpoint_hashes'][path.name]
    train_full_day(model_class=SlowMemoryModel,precompute_fn=cache,epochs=240,
        checkpoints=(0,80,240),warm_start_dir=parent,warm_start_epoch=80,learning_rate=.0003)
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result.update({'warm_start_epoch':80,'additional_epochs':240,
        'optimizer_state':'fresh Adam, lr .0003','state_row_sum_cap':SlowMemoryModel.cap,
        'selected_total_epoch':80+result['selected_epoch'],
        'parent_summary_sha256':hashlib.sha256((parent/'summary.json').read_bytes()).hexdigest()})
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
