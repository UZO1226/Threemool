"""e67 사전 검사: w = 0 이면 e64 의 공기 0 + park_e19 와 총 공기만 PPO seed 0 결과 (DO 수정 모델, 일별, 상한) 를 정확히 재현한다."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
E64 = Path('/home/jio/fcsrl_src/results/e64/main/runs/corrected_cap1_daily')
KEYS = ('air_m3', 'cost', 'nh4_cost', 'do_cost', 'clipped_pct')


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    C = {}
    for tag, args, ref in (('C1', ['--base', 'zero'], 'zero'), ('C2', ['--base', 'ppo_total', '--seed', '0'], 'total_ppo_s0')):
        subprocess.run([sys.executable, str(HERE / 'evaluate67.py')] + args + ['--w', '0', '--model', 'corrected', '--out', str(a.out / tag)], check=True)
        new = json.loads((a.out / tag / 'result.json').read_text())['val']
        old = json.loads((E64 / ref / 'result.json').read_text())['val']
        C[tag] = dict(new={k: new[k] for k in KEYS}, old={k: old[k] for k in KEYS}, passed=all(new[k] == old[k] for k in KEYS))
    C['passed'] = C['C1']['passed'] and C['C2']['passed']
    (a.out / 'checks.json').write_text(json.dumps(C, indent=1))
    print(json.dumps(C, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
