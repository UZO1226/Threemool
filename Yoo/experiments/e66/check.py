"""e66 사전 검사. test 는 건드리지 않는다.

C1 evaluate66 을 --split val 로 부르면 e64 결과 (alloc PPO seed 0, DO 수정 모델, 일별, 상한) 와 같다.
C2 연속 모드도 e64 결과 (같은 정책, 연속, 상한) 와 같다.
C3 test 날짜 41일이 연속이며 train, val 과 겹치지 않는다. 날짜 목록만 읽고 상태나 비용은 계산하지 않는다.
C4 잠금 파일이 아직 없다 (test 미사용).
C5 e65 정책 20개의 model_final 이 모두 있다 (e65 완료 뒤 실행).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate66 as e66  # noqa: E402
ev, base = e66.ev, e66.base
E64 = Path('/home/jio/fcsrl_src/results/e64/main/runs')
E65 = Path('/home/jio/fcsrl_src/results/e65/main/runs')
KEYS = ('air_m3', 'cost', 'nh4_cost', 'do_cost', 'clipped_pct', 'w_mean')


def hashes():
    files = [HERE / 'evaluate66.py', HERE / 'run.py', HERE / 'report.py', HERE.parent / 'e64/evaluate64.py', HERE.parent / 'e63/alloc.py',
             HERE.parent / 'e59/min_brake.py', HERE.parent / 'e59/shield_eval.py', HERE.parent / 'e30/darrow_env.py', HERE.parent / 'e23/behavior.py',
             ev.PREPARED / 'data.pkl', ev.MODELS['corrected'], ev.MODELS['common']]
    files += sorted(p for p in E65.glob('*/*/seed_*/model_final/*'))
    files += sorted(p for p in ev.E63.glob('*/*/seed_*/model_final/*'))
    return {str(p): base.sha(p) for p in files}


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    C = {}
    for tag, mode, name in (('C1', 'daily', 'corrected_cap1_daily'), ('C2', 'continuous', 'corrected_cap1_continuous')):
        out = a.out / tag
        subprocess.run([sys.executable, str(HERE / 'evaluate66.py'), '--split', 'val', '--policy', 'policy', '--condition', 'alloc', '--method', 'ppo',
                        '--seed', '0', '--model', 'corrected', '--cap', '1', '--mode', mode, '--out', str(out)], check=True)
        new = json.loads((out / 'result.json').read_text())['val']
        old = json.loads((E64 / name / 'alloc_ppo_s0' / 'result.json').read_text())['val']
        C[tag] = dict(new={k: new[k] for k in KEYS}, old={k: old[k] for k in KEYS}, passed=all(new[k] == old[k] for k in KEYS))
    d = pd.read_pickle(ev.PREPARED / 'data.pkl')
    test = pd.to_datetime(sorted(d['days']['test']))
    other = set(d['days']['train']) | set(d['days']['val'])
    C['C3'] = dict(days=len(test), first=str(test[0].date()), last=str(test[-1].date()),
                   passed=bool(len(test) == 41 and (pd.Series(test).diff().dropna() == pd.Timedelta(days=1)).all() and not (set(d['days']['test']) & other)))
    C['C4'] = dict(lock_exists=e66.LOCK.exists(), passed=not e66.LOCK.exists())
    missing = [f'{c}/{m}/seed_{s}' for c in ('total', 'alloc') for m in ('ppo', 'fcsrl') for s in range(5, 10)
               if not (E65 / c / m / f'seed_{s}' / 'model_final' / 'model.pt').exists()]
    C['C5'] = dict(missing=missing, passed=not missing)
    C['passed'] = all(C[k]['passed'] for k in ('C1', 'C2', 'C3', 'C4', 'C5'))
    C['hashes'] = hashes()
    base.save(a.out / 'checks.json', C)
    print(json.dumps({k: v for k, v in C.items() if k != 'hashes'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
