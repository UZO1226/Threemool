"""직접 만든 DARROW 환경이 Lee 의 환경과 같은 값을 내는지 검사한다 (Park 작성, Park/e08).

실제 운전 공기량을 재생한 학습, 검증, 평가 구간 지표를 두 가지와 비교한다.
1. 같은 대리 모델 pickle 로 Lee 의 calibrate_dw.py 를 돌린 결과 (완전히 같아야 한다)
2. Lee/experiments/e07/README.md 에 적힌 값 (하루 비용 학습 12.17, 검증 28.77, 평가 8.80, 한도 12)

    python check_darrow.py --data ../data/darrow/darrow_15min.pkl --surrogate ../data/darrow/darrow_surrogate_gbm.pkl \
        --calibration ../data/darrow/darrow_calibration.json --out ../experiments/e08/check_darrow.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from darrow_env import BatchEnv, Dynamics, evaluate, load_days

KEYS = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'n_episodes')
LEE_README_COST = {'train': 12.17, 'val': 28.77, 'test': 8.80}


def main():
    p = argparse.ArgumentParser()
    for k in ('data', 'surrogate', 'calibration', 'out'):
        p.add_argument(f'--{k}', type=Path, required=True)
    a = p.parse_args()
    splits, day_data = load_days(a.data)
    env = BatchEnv(day_data, Dynamics(a.surrogate))
    ref = json.loads(a.calibration.read_text())
    report, ok = {}, True
    for split in ('train', 'val', 'test'):
        mine = evaluate(env, splits[split], lambda obs: env.historical_action())
        # Lee/code/darrow/env_dw.py 의 step 은 하루가 끝난 환경을 reset 할 때 prev_air[i] 를 다음 에피소드 첫 공기량
        # 으로 덮어쓰는데, 이 배열이 그 스텝의 info['air'] 와 같은 객체다. 그래서 Lee 의 공기량 합계는 마지막 스텝을
        # 다음 에피소드(재생 평가에서는 같은 날) 의 첫 기록 공기량으로 센다. 그 차이를 반영한 값과 비교한다.
        last_minus_first = np.mean([day_data[d]['air'][env.T - 1] - day_data[d]['air'][0]
                                    for d in splits[split] for _ in range(env.dyn.n_members)])
        mine['air_m3_lee_counting'] = mine['air_m3'] - 0.25 * last_minus_first
        rows = []
        for k in KEYS:
            mk = 'air_m3_lee_counting' if k == 'air_m3' else k
            same = bool(abs(mine[mk] - ref[split][k]) <= 1e-6 * max(1.0, abs(ref[split][k])))
            ok &= same
            rows.append({'metric': mk, 'value': mine[mk], 'lee_code_same_model': ref[split][k], 'passed': same})
            print(f'{split:5s} {mk:20s} {mine[mk]:12.4f}  Lee 코드 {ref[split][k]:12.4f}  {"통과" if same else "실패"}')
        rows.append({'metric': 'air_m3_correct_counting', 'value': mine['air_m3']})
        rows.append({'metric': 'cost_vs_lee_readme', 'value': round(mine['cost'], 2), 'lee_readme': LEE_README_COST[split]})
        print(f'{split:5s} 하루 비용 {mine["cost"]:.2f}  Lee README {LEE_README_COST[split]:.2f}')
        report[split] = rows
        if split == 'train':
            train_cost = mine['cost']
    report['all_passed_same_model'] = bool(ok)
    report['cost_limit'] = float(round(train_cost))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('비용 한도', report['cost_limit'], '| Lee 코드와', '전체 통과' if ok else '실패한 항목이 있다')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
