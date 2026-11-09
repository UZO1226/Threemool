"""Park/e07 Kaggle Avedøre 에서 실행 시점 안전 차단 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e07.py one --seed 0 --data <aved_5min.pkl> --runs <결과>/av_runs --out <결과>/e07
    python run_e07.py report --out <결과>/e07 --runs <결과>/av_runs --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e03'))
from agent import FCSRL, ObsNormalizer  # noqa: E402
from avedore_env import (BatchEnv, NH4_MAX, Surrogate, action_to_air, air_to_action, all_episodes,  # noqa: E402
                         evaluate, load_days)
from run_e03 import Logistic  # noqa: E402
from train_avedore import SURROGATE, agent_act, avedore_config  # noqa: E402

BRAKE_AIR = 5400.0
HORIZON = 12
N_FIT_EPISODES = 300
CURVES = {'nh4_rule': (2.0, 2.5, 3.0, 3.5, 4.0),
          'onoff': (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
          'graded': (0.1, 0.25, 0.5, 1.0)}
LABELS = {'nh4_rule': 'NH4 규칙 theta', 'onoff': '예측기 켜고 끄기 기준', 'graded': '예측기 비례 c'}
COMPARISONS = (('H1', 'onoff', 'nh4_rule'), ('H2', 'graded', 'onoff'))
KEYS = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')


def features(obs):
    obs = np.asarray(obs, float)
    angle = np.arctan2(obs[:, 7], obs[:, 8])     # Avedøre 관측 7, 8번이 하루 중 시각 sin, cos
    return np.concatenate([obs] + [f(k * angle)[:, None] for k in (2, 3) for f in (np.sin, np.cos)], axis=1)


def fit_predictor(agent, env, pool, seed):
    rng = np.random.default_rng(7000 + seed)
    eps = [pool[i] for i in rng.choice(len(pool), N_FIT_EPISODES, replace=False)]
    obs = env.reset(eps)
    O, V, done = [], [], False
    while not done:
        O.append(obs)
        obs, _, _, done, info = env.step(agent_act(agent, obs))
        V.append(info['v_nh4'])
    O, V = np.stack(O), np.stack(V)               # (T, N, 11), (T, N). V[t] 는 스텝 t 뒤 NH4 위반
    T = len(V)
    x = O[:T - HORIZON + 1].reshape(-1, O.shape[-1])
    y = np.stack([V[t:t + HORIZON].any(0) for t in range(T - HORIZON + 1)]).reshape(-1).astype(float)
    model = Logistic().fit(features(x), y)
    return model, float(y.mean())


def brake(trigger_air):
    """trigger_air(obs) -> 차단 공기량 배열 (0 이면 차단 안 함). 실행 공기량 = max(정책, 차단)."""
    def shield(obs, a, env):
        phase = env.phase()
        policy_air = action_to_air(a, phase)
        target = np.where(phase == 2, trigger_air(obs), 0.0)
        air = np.maximum(policy_air, target)
        used = air > policy_air + 1e-9
        return np.where(used, air_to_action(air), a), used
    return shield


def run_one(a):
    splits, day_data = load_days(a.data)
    sur = Surrogate(SURROGATE)
    env = BatchEnv(day_data, sur)
    run = Path(a.runs) / f'fcsrl_s{a.seed}' / 'model_final'
    agent = FCSRL(11, 1, avedore_config(), ObsNormalizer(np.zeros((2, 11))))
    agent.load(run)
    model, pos = fit_predictor(agent, env, all_episodes(splits['train'], sur.n_members), a.seed)

    def prob(obs):
        return 1 / (1 + np.exp(-np.clip(model.score(features(obs)), -30, 30)))

    act = lambda obs: agent_act(agent, obs)  # noqa: E731
    makers = {'nh4_rule': lambda v: brake(lambda o: np.where(o[:, 0] >= v, BRAKE_AIR, 0.0)),
              'onoff': lambda v: brake(lambda o: np.where(prob(o) >= v, BRAKE_AIR, 0.0)),
              'graded': lambda v: brake(lambda o: BRAKE_AIR * np.minimum(1.0, prob(o) / v))}
    out = {'seed': a.seed, 'fit_positive_rate': pos, 'none': evaluate(env, splits['val'], act)}
    for kind, values in CURVES.items():
        out[kind] = {str(v): evaluate(env, splits['val'], act, shield=makers[kind](v)) for v in values}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'seed_{a.seed}.json').write_text(json.dumps(out, indent=2))
    print('완료', a.seed, flush=True)


def gaps(front, ref):
    ref = sorted(ref.values(), key=lambda p: p['air_m3'])
    xs, ys = [p['air_m3'] for p in ref], [p['cost'] for p in ref]
    return [p['cost'] - float(np.interp(p['air_m3'], xs, ys)) for p in front.values() if xs[0] <= p['air_m3'] <= xs[-1]]


def ms(v, d=2):
    v = np.asarray(v, float)
    return f'{v.mean():.{d}f} ± {v.std():.{d}f}'


def report(a):
    res = {json.loads(p.read_text())['seed']: json.loads(p.read_text()) for p in Path(a.out).glob('seed_*.json')}
    seeds = sorted(res)
    L = [f'FCSRL + PPO-Lag 정책 seed {len(seeds)}개, 평균 ± 표준편차(ddof=0). 검증 43일 x 구성원 5개 = 215 에피소드.', '',
         '## 단계 1: 학습된 정책 (차단 없음, 마지막 정책)', '',
         '| 방법 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 만족 % |', '| --- | --- | --- | --- | --- | --- |']
    for m, label in (('ppo', 'PPO-Lag'), ('fcsrl', 'FCSRL + PPO-Lag')):
        ms_ = [json.loads((Path(a.runs) / f'{m}_s{s}' / 'metrics.json').read_text())['final_val'] for s in seeds
               if (Path(a.runs) / f'{m}_s{s}' / 'metrics.json').exists()]
        if ms_:
            L.append(f"| {label} | {ms([x['air_m3'] for x in ms_], 0)} | {ms([x['cost'] for x in ms_])} | "
                     f"{ms([x['nh4_viol_pct'] for x in ms_])} | {ms([x['do_viol_pct'] for x in ms_])} | "
                     f"{ms([x['days_within_limit_pct'] for x in ms_], 1)} |")
    L += ['', '## 단계 2: 차단 곡선 (FCSRL 정책 평균)', '',
          '| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 만족 % | 개입 % |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    rs = [res[s] for s in seeds]
    row = lambda label, ps: (f"| {label} | {ms([p['air_m3'] for p in ps], 0)} | {ms([p['cost'] for p in ps])} | "  # noqa: E731
                             f"{ms([p['nh4_viol_pct'] for p in ps])} | {ms([p['do_viol_pct'] for p in ps])} | "
                             f"{ms([p['days_within_limit_pct'] for p in ps], 1)} | {ms([p.get('intervention_pct', 0) for p in ps], 1)} |")
    L.append(row('차단 없음', [r['none'] for r in rs]))
    for kind, values in CURVES.items():
        for v in values:
            L.append(row(f'{LABELS[kind]} {v}', [r[kind][str(v)] for r in rs]))
    L += ['', '## 같은 공기량 차이 (앞 곡선 하루 비용 빼기 비교 곡선 보간 하루 비용, 음수면 앞 곡선이 낫다)', '',
          '| seed | 켜고 끄기 대 NH4 규칙 (점) | 비례 대 켜고 끄기 (점) | theta 3.0 하루 비용 빼기 차단 없음 |',
          '| --- | --- | --- | --- |']
    table = {}
    for s in seeds:
        cells = []
        for hid, front, ref in COMPARISONS:
            g = gaps(res[s][front], res[s][ref])
            table[(hid, s)] = float(np.mean(g)) if g else None
            cells.append(f"{'없음' if not g else f'{np.mean(g):.2f}'} ({len(g)})")
        d3 = res[s]['nh4_rule']['3.0']['cost'] - res[s]['none']['cost']
        table[('H3', s)] = d3
        L.append(f'| {s} | ' + ' | '.join(cells) + f' | {d3:.2f} |')
    L += ['', '## 사전 등록 판정 (seed 3개 중 2개 이상 음수, 평균 음수)', '']
    for hid, label in (('H1', '켜고 끄기 대 NH4 규칙'), ('H2', '비례 대 켜고 끄기'), ('H3', 'theta 3.0 대 차단 없음')):
        vals = [table[(hid, s)] for s in seeds]
        known = [v for v in vals if v is not None]
        neg = sum(v is not None and v < 0 for v in vals)
        ok = neg >= 2 and bool(known) and np.mean(known) < 0
        mean_txt = f'{np.mean(known):.2f}' if known else '없음'
        L.append(f'- {hid} ({label}): 평균 {mean_txt}, 음수 seed {neg}/{len(seeds)}: {"지지" if ok else "지지 안 됨"}')
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    o.add_argument('--seed', type=int, required=True)
    for k in ('data', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'runs', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
