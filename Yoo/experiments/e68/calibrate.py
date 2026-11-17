"""e68 0단계: dev 학습 날짜에서만 하루 비용의 범위를 잰다. 한도 d 를 정하는 근거이며 val, holdout 은 읽지 않는다.

기록 운전 재생, 고정 공기 0 ~ 9,000 (500 간격) 을 대리 모델 구성원 5개로 돈다.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env68 as E  # noqa: E402
from darrow_env import air_to_action  # noqa: E402


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    splits, days = E.load()
    cap = E.caps(days, splits['dev'])
    env = E.DailyEnv(days, E.Dynamics(E.PREPARED / 'surrogate.pkl'), cap)
    out = dict(cap=cap.tolist(), dev_days=len(splits['dev']), rows={})
    res, *_ = E.rollout(env, splits['dev'], lambda o: np.clip(air_to_action(env.x['air'][:, env.t]), -1, 1))
    out['rows']['record'] = res
    print('record', json.dumps(res))
    for air in range(0, 9001, 500):
        res, *_ = E.rollout(env, splits['dev'], lambda o, air=air: np.full(len(o), air_to_action(air)))
        out['rows'][f'const_{air}'] = res
        print(air, round(res['air_m3']), round(res['day_violation_pct'], 2), round(res['nh4_mean'], 2), round(res['nh4_step_viol_pct'], 2), round(res['do_step_viol_pct'], 2))
    obs = np.concatenate([days[d]['state'][1:, 0] for d in splits['dev']])
    out['measured_dev'] = dict(nh4_mean=float(obs.mean()), day_violation_pct=float(100 * np.mean([days[d]['state'][1:, 0].mean() > 4 for d in splits['dev']])))
    (a.out / 'calibration.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
