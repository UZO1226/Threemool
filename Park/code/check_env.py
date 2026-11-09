"""직접 만든 환경이 팀 기준값을 재현하는지 검사한다 (Park 작성).

기준값은 Lee/experiments/e01/results.md 의 기준 제어 표(건기, 14일 운전의 후반 7일) 다.
모두 통과해야 학습으로 넘어간다.

    export BSM1_ARCHIVE=<BSM1.zip 경로>
    python check_env.py --prepared <결과 폴더>/prepared --out ../experiments/e01/check_env.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from baselines import BASELINES
from evaluate import evaluate

# (폭기 kWh/day, 유량 가중 NH4, 순간 위반 %, 하루 위반 스텝)
TARGETS_DRY = {
    'fixed84': (3341.4, 4.831, 63.24, 182.1),
    'fixed240': (4265.6, 2.308, 15.82, 45.6),
    'do_pi': (3717.8, 2.567, 17.11, 49.3),
}
KEYS = ('aeration_kwh_per_day', 'flow_weighted_nh4', 'step_violation_percent', 'violating_steps_per_day')
# 표의 반올림 자릿수에 맞춘 허용 오차. 이보다 크면 환경 정의가 다른 것으로 본다.
TOLERANCE = (0.15, 0.0015, 0.015, 0.15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    initial = dict(np.load(a.prepared / 'initial.npz'))

    report, ok = {}, True
    for name, (policy, controller) in BASELINES.items():
        result = evaluate(initial, policy, controller=controller)
        dry = result['dry']['week']
        rows = []
        for key, target, tol in zip(KEYS, TARGETS_DRY[name], TOLERANCE):
            value = dry[key]
            passed = abs(value - target) <= tol
            ok &= passed
            rows.append({'metric': key, 'value': value, 'target': target, 'passed': passed})
            print(f'{name:9s} {key:26s} {value:10.3f}  기준 {target:10.3f}  {"통과" if passed else "실패"}')
        report[name] = {'dry_check': rows, 'dry': dry, 'rain': result['rain']['week']}

    report['all_passed'] = bool(ok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('전체 통과' if ok else '실패한 항목이 있다. 학습으로 넘어가지 않는다')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
