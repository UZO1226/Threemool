"""e51: 공통 기준 대리 모델(scikit-learn 1.9.1)의 DO 구간 대응 오류가 팀 환경 수치에 주는 영향

e35 에서 확인한 대응(DO1/2/3 은 공기 구간 3/4/5)으로 DO 모델 15개만 다시 학습하고, NH4 와 NO3 모델은 그대로 둔다.
e35 절차와 같지만, 바꾸지 않은 모델의 동일성은 pickle 바이트 대신 예측값으로 확인한다 (1.9.1 에서는 같은 모델도
pickle 바이트가 달라진다). 정책 학습은 하지 않고 test 구간은 쓰지 않는다.

비교 항목
1. 기록 공기량 재생의 학습, 검증 하루 비용과 NH4, DO 위반 (PROTOCOL 검산값과 같은 지표)
2. e35 와 같은 장기 재생 오차와 위반 탐지 (e35 evaluate 재사용)
3. 기록 총 공기량에 배율(0.5 ~ 1.5)을 곱했을 때 검증 구간 비용과 위반: 정책이 공기를 줄일 때 환경이 어떻게 답하는지

    python run.py --prepared /home/jio/fcsrl_src/results/e50/prepared --out /home/jio/fcsrl_src/results/e51/main
"""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
sys.path.insert(0, str(HERE.parent / 'e35'))
from common import save, sha  # noqa: E402
from darrow_env import BatchEnv, Dynamics, load_days, all_episodes, STEP_H, NH4_MAX, DO_MAX  # noqa: E402
from mapping import monotone_constraints, legacy_constraints, verify, STATE  # noqa: E402
from surrogate_dw import build_arrays  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e35 = load_module('e35_run', HERE.parent / 'e35' / 'run.py')
SCALES = (.5, .75, 1., 1.25, 1.5)


def replay(env, days, scale=1.):
    env.reset(all_episodes(days, env.dyn.n_members))
    N = len(env.episodes)
    cost, vn, vd, air = (np.zeros(N) for _ in range(4))
    done = False
    while not done:
        act = env.historical_action()
        if scale != 1.:
            act = np.clip((act + 1) * scale - 1, -1, 1)
        _, _, c, done, info = env.step(act)
        cost, vn, vd, air = cost + c, vn + info['v_nh4'], vd + info['v_do'], air + info['air'] * STEP_H
    return {'cost': float(cost.mean()), 'nh4_pct': float(100 * vn.mean() / env.T), 'do_pct': float(100 * vd.mean() / env.T),
            'air_m3': float(air.mean())}


def main(a):
    start = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=False)
    verify()
    old_path = a.prepared / 'surrogate.pkl'
    old = pickle.loads(old_path.read_bytes())
    data = pd.read_pickle(a.prepared / 'data.pkl')
    x, y = build_arrays(data['data'], data['days']['train'])
    xv, _ = build_arrays(data['data'], data['days']['val'])
    for row in old['models']:
        for j, model in enumerate(row):
            assert np.array_equal(model.monotonic_cst, legacy_constraints(j))
    fixed = {'models': [list(row) for row in old['models']], 'hi': old['hi']}
    replica_exact = None
    for m in range(5):
        ids = np.random.RandomState(m).choice(len(x), int(.8 * len(x)), replace=False)
        for j in (2, 3, 4):
            params = old['models'][m][j].get_params()
            if m == 0 and j == 2:
                replica = HistGradientBoostingRegressor(**params).fit(x[ids], y[ids, j])
                replica_exact = bool(np.array_equal(replica.predict(xv), old['models'][m][j].predict(xv)))
                assert replica_exact, '기존 모델을 같은 절차로 재현하지 못했다'
            params['monotonic_cst'] = monotone_constraints(j)
            fixed['models'][m][j] = HistGradientBoostingRegressor(**params).fit(x[ids], y[ids, j])
            print('FIT', m, STATE[j], f'{time.monotonic() - start:.0f}s', flush=True)
    for m in range(5):
        for j in (0, 1):
            assert fixed['models'][m][j] is old['models'][m][j]
            assert np.array_equal(fixed['models'][m][j].predict(xv), old['models'][m][j].predict(xv))
    new_path = a.out / 'corrected_surrogate.pkl'
    new_path.write_bytes(pickle.dumps(fixed))
    splits, days = load_days(a.prepared / 'data.pkl')
    out = {'replica_prediction_exact': replica_exact, 'input_sha256': {'surrogate.pkl': sha(old_path)},
           'corrected_sha256_not_identity': sha(new_path), 'test_evaluated': False, 'policy_training': False}
    for name, path in (('legacy', old_path), ('corrected', new_path)):
        env = BatchEnv(days, Dynamics(path))
        r = {'replay': {s: replay(env, splits[s]) for s in ('train', 'val')},
             'air_scale_val': {str(k): replay(env, splits['val'], k) for k in SCALES}}
        r['e35_metrics'] = e35.evaluate(a.prepared, path)[0]
        out[name] = r
        print('DONE', name, json.dumps(r['replay']), flush=True)
    save(a.out / 'results.json', out)
    md = ['| 지표 | 기준 모델 (기존 대응) | DO 대응 수정 |', '| --- | --- | --- |']
    for s in ('train', 'val'):
        for k, lab in (('cost', '하루 비용'), ('nh4_pct', 'NH4 위반 %'), ('do_pct', 'DO 위반 %')):
            md.append(f"| {s} 재생 {lab} | {out['legacy']['replay'][s][k]:.2f} | {out['corrected']['replay'][s][k]:.2f} |")
    for key in STATE:
        v = [out[c]['e35_metrics']['rollout_member_average']['mae'][key] for c in ('legacy', 'corrected')]
        md.append(f'| val {key} 장기 재생 MAE | {v[0]:.3f} | {v[1]:.3f} |')
    for nm in ('NH4', 'DO'):
        for k in ('recall_pct', 'fpr_pct'):
            v = [out[c]['e35_metrics']['rollout_member_average']['confusion'][nm][k] for c in ('legacy', 'corrected')]
            md.append(f'| val {nm} {k} | {v[0]:.2f} | {v[1]:.2f} |')
    for t in ('do1', 'do2', 'do3'):
        v = [np.mean(out[c]['e35_metrics']['physical_zone_wrong_sign_pct_by_member'][t]) for c in ('legacy', 'corrected')]
        md.append(f'| {t}: 직접 구간 공기 증가에 DO 감소 비율 % | {v[0]:.2f} | {v[1]:.2f} |')
    md += ['', '| 검증 기록 총 공기량 배율 | 기준 모델 비용 (NH4 %, DO %) | 수정 모델 비용 (NH4 %, DO %) |', '| --- | --- | --- |']
    for k in SCALES:
        cell = [out[c]['air_scale_val'][str(k)] for c in ('legacy', 'corrected')]
        md.append(f"| {k} | " + ' | '.join(f"{v['cost']:.2f} ({v['nh4_pct']:.2f}, {v['do_pct']:.2f})" for v in cell) + ' |')
    (a.out / 'results.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
