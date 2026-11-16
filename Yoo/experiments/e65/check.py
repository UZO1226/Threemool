"""e65 사전 검사.

C1 e63 사전 검사에서 고정한 학습 코드와 준비물 해시가 지금도 같다 (e63 학습 코드 불변).
C2 runs-root 를 받도록 고친 e64 평가 코드가 e64 결과 하나 (alloc PPO seed 0, DO 수정 모델, 일별, 상한) 를 정확히 재현한다.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e64'))
import evaluate64 as ev  # noqa: E402
base = ev.base
E63_CHECK = Path('/home/jio/fcsrl_src/results/e63/check/checks.json')
E64_RESULT = Path('/home/jio/fcsrl_src/results/e64/main/runs/corrected_cap1_daily/alloc_ppo_s0/result.json')


def hashes():
    h = dict(json.loads(E63_CHECK.read_text())['hashes'])
    for p in (HERE.parent / 'e64/evaluate64.py', HERE / 'run.py', HERE / 'report.py'):
        h[str(p)] = base.sha(p)
    return h


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    e63 = json.loads(E63_CHECK.read_text())
    changed = [p for p, v in e63['hashes'].items() if base.sha(p) != v]
    C = {'C1': dict(changed=changed, passed=not changed and e63['passed'])}
    out = a.out / 'c2'
    subprocess.run([sys.executable, str(HERE.parent / 'e64/evaluate64.py'), '--policy', 'policy', '--condition', 'alloc', '--method', 'ppo', '--seed', '0',
                    '--model', 'corrected', '--cap', '1', '--mode', 'daily', '--shield', 'park_e19', '--runs-root', str(ev.E63), '--out', str(out)], check=True)
    new = json.loads((out / 'result.json').read_text())['val']
    old = json.loads(E64_RESULT.read_text())['val']
    keys = ('air_m3', 'cost', 'nh4_cost', 'do_cost', 'clipped_pct', 'w_mean')
    C['C2'] = dict(new={k: new[k] for k in keys}, old={k: old[k] for k in keys}, passed=all(new[k] == old[k] for k in keys))
    C['passed'] = C['C1']['passed'] and C['C2']['passed']
    C['hashes'] = hashes()
    base.save(a.out / 'checks.json', C)
    print(json.dumps({k: v for k, v in C.items() if k != 'hashes'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
