"""직접 만든 Avedøre 환경이 Yoo 의 기록을 재현하는지 검사한다 (Park 작성, Park/e07).

실제 운전 공기량을 그대로 재생했을 때의 지표를 `Yoo/experiments/e02/historical.json` 과 비교한다.
모두 같아야 학습으로 넘어간다.

    python check_avedore.py --data ../data/avedore/aved_5min.pkl --out ../experiments/e07/check_avedore.json
"""
import argparse
import json
from pathlib import Path

from avedore_env import BatchEnv, Surrogate, evaluate, load_days

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SURROGATE = ROOT / 'Yoo/experiments/e02/surrogate.json'
KEYS = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'n_episodes')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    splits, day_data = load_days(a.data)
    env = BatchEnv(day_data, Surrogate(SURROGATE))
    target = json.loads((ROOT / 'Yoo/experiments/e02/historical.json').read_text())
    report, ok = {}, True
    for split in ('val', 'test'):
        mine = evaluate(env, splits[split], lambda obs: env.historical_action())
        rows = []
        for k in KEYS:
            same = abs(mine[k] - target[split][k]) <= 1e-6 * max(1.0, abs(target[split][k]))
            ok &= same
            rows.append({'metric': k, 'value': mine[k], 'target': target[split][k], 'passed': same})
            print(f'{split:5s} {k:24s} {mine[k]:14.6f}  Yoo {target[split][k]:14.6f}  {"통과" if same else "실패"}')
        report[split] = rows
    report['all_passed'] = bool(ok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('전체 통과' if ok else '실패한 항목이 있다')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
