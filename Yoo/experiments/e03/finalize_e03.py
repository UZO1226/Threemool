"""35개 실행 완료 확인 후 재평가와 표 생성을 순서대로 수행한다. 학습은 시작하지 않는다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[2] / 'code' / 'wwtp'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_runs(e02, e03):
    return [root / tag / f'seed_{seed}' for root, tags in (
        (e02, ('td3', 'td3lag', 'fcsrl')),
        (e03, ('fcsrl_ucb', 'ppolag', 'fcsrl_ml5', 'fcsrl_ucb_ml5')))
        for tag in tags for seed in range(5)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('e02', 'e03', 'data', 'mlp', 'gbm', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--cudaid', type=int, default=-1)
    ap.add_argument('--wait', action='store_true')
    ap.add_argument('--timeout', type=float, default=7200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = expected_runs(a.e02, a.e03)
    deadline = time.monotonic() + a.timeout
    last_missing = None
    while True:
        missing = [str(p) for p in runs if not (p / 'metrics.json').is_file()]
        if not missing:
            break
        if missing != last_missing:
            print(f'{datetime.now().isoformat()} completed={len(runs) - len(missing)}/35, waiting={missing}', flush=True)
            last_missing = missing
        if not a.wait or time.monotonic() >= deadline:
            raise RuntimeError(f'Incomplete runs: {missing}')
        time.sleep(min(30, max(0, deadline - time.monotonic())))

    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'runs': [], 'inputs': {}, 'source': {}}
    for p in (a.data, a.mlp, a.gbm):
        manifest['inputs'][str(p.resolve())] = digest(p)
    for p in sorted(CODE.glob('*.py')):
        manifest['source'][p.name] = digest(p)
    for run in runs:
        m = json.loads((run / 'metrics.json').read_text())
        best = run / 'model' / f"epoch_{m['best_epoch']}"
        artifacts = [run / 'metrics.json', best / 'model.pt']
        for name in ('normalizer.json', 'normalizer.stats', 'config.json'):
            if (best / name).exists():
                artifacts.append(best / name)
        if (run / 'model_final').is_dir():
            artifacts += sorted(p for p in (run / 'model_final').iterdir() if p.is_file())
        manifest['runs'].append({'tag': m.get('tag', m['method']), 'seed': m['seed'],
                                 'artifacts': {str(p.resolve()): digest(p) for p in artifacts}})
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for label, surrogate, checkpoint in (('mlp', a.mlp, 'best'), ('gbm', a.gbm, 'best'),
                                         ('final_gbm', a.gbm, 'final')):
        cmd = [sys.executable, str(CODE / 'evaluate.py'), 'crosseval', '--runs', str(a.e02), str(a.e03),
               '--data', str(a.data), '--surrogate', str(surrogate), '--checkpoint', checkpoint,
               '--cudaid', str(a.cudaid), '--out', str(a.out / f'{label}.json')]
        print(f'{datetime.now().isoformat()} evaluating {label}', flush=True)
        with (a.out / f'{label}.log').open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    cmd = [sys.executable, str(Path(__file__).with_name('summarize_e03.py')),
           '--mlp', str(a.out / 'mlp.json'), '--gbm', str(a.out / 'gbm.json'),
           '--final_gbm', str(a.out / 'final_gbm.json'), '--out', str(a.out)]
    with (a.out / 'summary.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (a.out / 'completion.json').write_text(json.dumps({
        'completed_at': datetime.now(timezone.utc).isoformat(), 'n_runs': len(runs),
        'artifacts': {p.name: digest(p) for p in a.out.glob('*.json') if p.name != 'completion.json'},
    }, indent=2))
    print(f'{datetime.now().isoformat()} COMPLETE: {a.out}', flush=True)


if __name__ == '__main__':
    main()
