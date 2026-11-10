"""e31 1단계 진단: DARROW 에서 실험 설계를 바꿀 근거를 학습 없이 잰다

e30 준비물(data.pkl, surrogate.pkl)을 그대로 읽는다. 평가(test) 구간은 쓰지 않는다.

A1 기록 배분 비율: 기록 총 공기량을 (a) 그 시각 기록 비율로 나눌 때와 (b) 학습 구간 중앙값 고정 비율로
   나눌 때 대리 모델 결과(비용, 위반)가 얼마나 다른가. 차이가 크면 기록 비율이 기존 제어기의 판단을 정책에
   공짜로 주는 셈이다
A2 계절 이동: 월별 실제 위반과 대리 모델 재생 위반, 수온 범위
A3 사건 예측 가능성: 지금 NH4 <= 4 인 시점에서 다음 2시간 안에 NH4 > 4 가 시작될지를 현재 관측으로
   맞히는 정도 (학습 구간으로 맞추고 검증 구간 AUROC). 입력 묶음별로 비교한다

    python diagnose.py --prepared /home/jio/fcsrl_src/results/e30/prepared --out .
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
from darrow_env import AIR_MAX, NH4_MAX, DO_MAX, STEP_H, BatchEnv, Dynamics, all_episodes, load_days  # noqa: E402


def replay(env, days, frac=None):
    """기록 총 공기량을 재생한다. frac 를 주면 배분 비율을 그 값으로 고정한다. 날짜별 평균(구성원 평균) 반환"""
    saved = {}
    if frac is not None:
        for d in days:
            saved[d] = env.day_data[d]['frac']
            env.day_data[d]['frac'] = np.broadcast_to(frac, saved[d].shape).copy()
    obs = env.reset(all_episodes(days, env.dyn.n_members))
    N = len(env.episodes)
    cost, vn, vd, air = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    done = False
    while not done:
        obs, _, c, done, info = env.step(env.historical_action())
        cost, vn, vd, air = cost + c, vn + info['v_nh4'], vd + info['v_do'], air + info['air']
    for d, f in saved.items():
        env.day_data[d]['frac'] = f
    per_day = pd.DataFrame({'day': [d for d, _ in env.episodes], 'cost': cost, 'nh4_pct': 100 * vn / env.T,
                            'do_pct': 100 * vd / env.T, 'air_m3': air * STEP_H}).groupby('day').mean()
    return per_day


def actual_per_day(day_data, days):
    rows = []
    for d in days:
        s = day_data[d]['state'][1:]
        vn, vd = s[:, 0] > NH4_MAX, s[:, 2:].max(1) > DO_MAX
        rows.append({'day': d, 'cost': float(vn.sum() + vd.sum()), 'nh4_pct': 100 * vn.mean(), 'do_pct': 100 * vd.mean(),
                     'air_m3': float(day_data[d]['air'][:-1].sum() * STEP_H)})
    return pd.DataFrame(rows).set_index('day')


def onset_dataset(day_data, days, horizon=8):
    """현재 NH4 <= 4 인 시점의 관측과 '다음 horizon 스텝 안에 NH4 > 4' 라벨. 하루 안에서만 본다"""
    X, y, ts = [], [], []
    for d in days:
        x = day_data[d]
        s, ex, air = x['state'], x['exo'], x['air']
        T = len(s)
        for t in range(1, T - horizon):
            if s[t, 0] > NH4_MAX:
                continue
            load = ex[t, 0] * ex[t, 4]
            X.append([s[t, 0], s[t - 1, 0], s[t, 1], *s[t, 2:], ex[t, 0], ex[t, 1], ex[t, 2], ex[t, 3], ex[t, 4], load,
                      air[t - 1]])
            y.append(float((s[t + 1:t + 1 + horizon, 0] > NH4_MAX).any()))
            ts.append(t)
    return np.array(X), np.array(y)


COLS = ['nh4', 'nh4_prev', 'no3', 'do1', 'do2', 'do3', 'q_in', 'temp', 'sin', 'cos', 'inf_nh4', 'load', 'air_prev']
SETS = {'현재 NH4 만': ['nh4', 'nh4_prev'], '시각만': ['sin', 'cos'], '유입(유량, 유입 NH4, 부하)': ['q_in', 'inf_nh4', 'load'],
        '유입 + 시각': ['q_in', 'inf_nh4', 'load', 'sin', 'cos'], '현재 NH4 + 유입': ['nh4', 'nh4_prev', 'q_in', 'inf_nh4', 'load'],
        '전체 관측': COLS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    splits, day_data = load_days(a.prepared / 'data.pkl')
    data = pd.read_pickle(a.prepared / 'data.pkl')
    median = np.array([data['zone_fractions'][f'air{i}'] for i in range(1, 6)])
    env = BatchEnv(day_data, Dynamics(a.prepared / 'surrogate.pkl'))
    out = {}

    # A1
    a1 = {}
    for split in ('train', 'val'):
        days = splits[split]
        rec, fix, act = replay(env, days), replay(env, days, median), actual_per_day(day_data, days)
        a1[split] = {k: {'cost': float(v.cost.mean()), 'nh4_pct': float(v.nh4_pct.mean()), 'do_pct': float(v.do_pct.mean()),
                         'air_m3': float(v.air_m3.mean())} for k, v in (('실측', act), ('재생_기록비율', rec), ('재생_고정비율', fix))}
        a1[split]['날짜별_비용차_고정_minus_기록'] = {'mean': float((fix.cost - rec.cost).mean()),
                                                 'abs_mean': float((fix.cost - rec.cost).abs().mean())}
        if split == 'val':
            per_day_val = pd.concat({'actual': act, 'replay_recorded': rec, 'replay_fixed': fix}, axis=1)
    out['A1_배분비율'] = a1
    per_day_val.to_csv(a.out / 'val_per_day_replay.csv')

    # A2
    rows = []
    for split in ('train', 'val'):
        rec, act = replay(env, splits[split]), actual_per_day(day_data, splits[split])
        for d in splits[split]:
            rows.append({'month': str(d)[:7], 'split': split, 'temp': float(day_data[d]['exo'][:, 1].mean()),
                         'actual_cost': act.cost[d], 'replay_cost': rec.cost[d], 'actual_nh4_pct': act.nh4_pct[d],
                         'replay_nh4_pct': rec.nh4_pct[d], 'q_in': float(day_data[d]['exo'][:, 0].mean()),
                         'inf_nh4': float(day_data[d]['exo'][:, 4].mean())})
    m = pd.DataFrame(rows).groupby(['month', 'split']).mean().round(2)
    out['A2_월별'] = m.reset_index().to_dict(orient='records')
    t_tr = [day_data[d]['exo'][:, 1].mean() for d in splits['train']]
    t_va = [day_data[d]['exo'][:, 1].mean() for d in splits['val']]
    out['A2_수온'] = {'train_일평균_범위': [float(min(t_tr)), float(max(t_tr))], 'val_일평균_범위': [float(min(t_va)), float(max(t_va))],
                    'val_중_train최저보다_추운날': int(sum(t < min(t_tr) for t in t_va))}

    # A3
    Xtr, ytr = onset_dataset(day_data, splits['train'])
    Xva, yva = onset_dataset(day_data, splits['val'])
    a3 = {'train_양성비율': float(ytr.mean()), 'val_양성비율': float(yva.mean()), 'n_train': int(len(ytr)), 'n_val': int(len(yva))}
    for name, cols in SETS.items():
        idx = [COLS.index(c) for c in cols]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[:, idx], ytr)
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(Xtr[:, idx], ytr)
        a3[name] = {'logistic_auroc': float(roc_auc_score(yva, lr.predict_proba(Xva[:, idx])[:, 1])),
                    'gbm_auroc': float(roc_auc_score(yva, gb.predict_proba(Xva[:, idx])[:, 1]))}
    out['A3_2시간내_위반시작_예측'] = a3
    (a.out / 'diagnostics.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != 'A2_월별'}, indent=1, ensure_ascii=False, default=float))
    print(m.to_string())


if __name__ == '__main__':
    main()
