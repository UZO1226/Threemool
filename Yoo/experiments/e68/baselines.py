"""e68 비교군: 기록 운전, 고정 공기, 암모니아 기반 폭기 제어 (ABAC).

ABAC 는 현장 표준인 NH4 피드백을 총 공기에 바로 거는 PI 제어기 (속도형) 다. 대리 모델이 공기를 직접 받으므로
NH4 -> DO 설정값 -> 공기의 연쇄를 NH4 -> 공기 한 단계로 줄였다.
  공기_t = clip(공기_{t-1} + Kp (e_t - e_{t-1}) + Ki e_t, 0, 9000),  e_t = NH4_t - 설정값
첫 스텝의 직전 공기는 그 시각 기록 공기, e_{-1} = e_0. 매개변수는 dev 학습 날짜에서만 격자로 고른다:
하루 예산 초과 스텝 <= 0.3 인 조합 중 공기가 가장 적은 것 (학습 한도와 같다). 고정 공기도 같은 규칙으로 100 간격에서 고른다.
설정값은 2.0 ~ 4.0 을 0.1 간격으로 본다. 고른 값이 격자 끝이면 실패로 멈춘다 (Kp 0 은 P 항 없음이라 허용).
Ki 는 400 ~ 25,600 과 1e6 이다. 1e6 은 설정값 위면 최대, 아래면 0 인 on/off 제어의 극한이라 Ki 최댓값은 끝으로 보지 않는다.
첫 시험에서 Ki 1,600 이 격자 끝에 걸렸고, 근처를 더 보니 Ki 를 키울수록 공기가 조금씩 줄어 이 격자로 넓혔다 (dev 날짜만 사용).
적분 포화 방지는 넣지 않았다. 검토에서 실제 공기 기준 포화 방지를 시험했을 때 공기는 줄고 위반이 늘어 더 강한 기준선이 아니었다.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env68 as E  # noqa: E402
from darrow_env import AIR_MAX, air_to_action  # noqa: E402

LIMIT = 0.3
SETPOINTS = (0.5, 1.0, 1.5) + tuple(round(2.0 + 0.1 * i, 1) for i in range(21))
KP = (0.0, 500.0, 1000.0, 2000.0)
KI = (400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0, 25600.0, 1e6)
CONSTANT = tuple(range(3000, 9001, 100))


class ABAC:
    def __init__(self, env, sp, kp, ki):
        self.env, self.sp, self.kp, self.ki = env, sp, kp, ki

    def __call__(self, obs):
        e = obs[:, 0].astype(float) - self.sp
        if self.env.t == 0:
            self.air, self.e = self.env.x['air'][:, 0].astype(float), e
        self.air = np.clip(self.air + self.kp * (e - self.e) + self.ki * e, 0.0, AIR_MAX)
        self.e = e
        return air_to_action(self.air)


def make_env():
    splits, days = E.load()
    return splits, E.DailyEnv(days, E.Dynamics(E.PREPARED / 'surrogate.pkl'), E.caps(days, splits['dev']))


def policy(env, kind, params):
    if kind == 'record':
        return lambda o: np.clip(air_to_action(env.x['air'][:, env.t]), -1, 1)
    if kind == 'constant':
        return lambda o: np.full(len(o), air_to_action(params['air']))
    return ABAC(env, params['sp'], params['kp'], params['ki'])


def _eval(job):
    os.environ['OMP_NUM_THREADS'] = '1'
    kind, params, split = job
    splits, env = make_env()
    res, *_ = E.rollout(env, splits[split], policy(env, kind, params))
    return kind, params, split, res


def select(rows):
    ok = [r for r in rows if r[3]['over_budget_steps'] <= LIMIT]
    return min(ok, key=lambda r: r[3]['air_m3']) if ok else None


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    jobs = [('abac', dict(sp=sp, kp=kp, ki=ki), 'dev') for sp, kp, ki in itertools.product(SETPOINTS, KP, KI)]
    jobs += [('constant', dict(air=float(x)), 'dev') for x in CONSTANT]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(_eval, jobs))
    grid = [dict(kind=k, params=p, dev=r) for k, p, _, r in rows]
    chosen = {}
    for kind in ('abac', 'constant'):
        best = select([r for r in rows if r[0] == kind])
        assert best is not None, f'{kind}: 한도를 만족하는 조합이 없다'
        chosen[kind] = best[1]
    (a.out / 'grid.json').write_text(json.dumps(grid, indent=0))
    p = chosen['abac']
    edge = [n for n, v, grid in (('sp', p['sp'], SETPOINTS), ('kp', p['kp'], KP), ('ki', p['ki'], KI))
            if (v == max(grid) and n != 'ki') or (v == min(grid) and n != 'kp')]
    assert not edge, f'ABAC 선택값이 격자 끝이다: {edge} {p}'
    assert CONSTANT[0] < chosen['constant']['air'] < CONSTANT[-1]
    chosen['record'] = {}
    final = {k: dict(params=p) for k, p in chosen.items()}
    with ProcessPoolExecutor(a.jobs) as ex:
        for kind, params, split, res in ex.map(_eval, [(k, p, s) for k, p in chosen.items() for s in ('dev', 'val')]):
            final[kind][split] = res
    (a.out / 'baselines.json').write_text(json.dumps(dict(limit=LIMIT, chosen=final, holdout_evaluated=False, test_evaluated=False), indent=1))
    for k, v in final.items():
        print(k, v['params'], {s: (round(v[s]['air_m3']), round(v[s]['day_violation_pct'], 2), round(v[s]['nh4_mean'], 2)) for s in ('dev', 'val')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=6)
    main(p.parse_args())
