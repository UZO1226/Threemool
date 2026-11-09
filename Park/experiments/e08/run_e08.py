"""Park/e08 팀 공식 데이터(Kaggle DARROW) 에서 실행 시점 안전 차단 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e08.py one --seed 0 --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과>/dw_runs --out <결과>/e08
    python run_e08.py report --out <결과>/e08 --runs <결과>/dw_runs --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e03'))
sys.path.insert(0, str(HERE.parent / 'e07'))
from agent import FCSRL, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, action_to_air, air_to_action, all_episodes, evaluate, load_days  # noqa: E402
from run_e03 import Logistic  # noqa: E402
from run_e07 import features, gaps, ms  # noqa: E402  (관측 7, 8번이 시각 sin, cos 인 점이 같다)
from train_darrow import agent_act, darrow_config  # noqa: E402

BRAKE_AIR = 5800.0
HORIZON = 4               # 15분 x 4 = 1시간
N_FIT_EPISODES = 300
SEEDS = (0, 1, 2, 3, 4)
CURVES = {'nh4_rule': (2.0, 2.5, 3.0, 3.5, 4.0),
          'onoff': (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
          'graded': (0.1, 0.25, 0.5, 1.0)}
LABELS = {'nh4_rule': 'NH4 규칙 theta', 'onoff': '예측기 켜고 끄기 기준', 'graded': '예측기 비례 c'}


def fit_predictor(agent, env, pool, seed):
    rng = np.random.default_rng(7000 + seed)
    eps = [pool[i] for i in rng.choice(len(pool), N_FIT_EPISODES, replace=False)]
    obs = env.reset(eps)
    O, V, done = [], [], False
    while not done:
        O.append(obs)
        obs, _, _, done, info = env.step(agent_act(agent, obs))
        V.append(info['v_nh4'])
    O, V = np.stack(O), np.stack(V)
    T = len(V)
    x = O[:T - HORIZON + 1].reshape(-1, O.shape[-1])
    y = np.stack([V[t:t + HORIZON].any(0) for t in range(T - HORIZON + 1)]).reshape(-1).astype(float)
    return Logistic().fit(features(x), y), float(y.mean())


def brake(trigger_air):
    def shield(obs, a, env):
        policy_air = action_to_air(a)
        air = np.maximum(policy_air, trigger_air(obs))
        used = air > policy_air + 1e-9
        return np.where(used, air_to_action(air), a), used
    return shield


def local_response(agent, env, days):
    """차단이 걸리는 순간(NH4 측정값 >= 3) 같은 상태에서 정책 공기량과 차단 공기량의 다음 스텝 NH4 비교."""
    obs = env.reset(all_episodes(days, env.dyn.n_members))
    d_pol, d_brk, done = [], [], False
    while not done:
        a = agent_act(agent, obs)
        trig = obs[:, 0] >= 3.0
        if trig.any():
            t, x = env.t, env.x
            air_p = action_to_air(a)[trig]
            frac = x['frac'][trig, t]
            nxt = lambda air: env.dyn.step(env.state[trig], x['exo'][trig, t], air[:, None] * frac,  # noqa: E731
                                            env.prev_zone[trig], env.member[trig])[:, 0]
            d_pol.append(nxt(air_p) - env.state[trig, 0])
            d_brk.append(nxt(np.maximum(air_p, BRAKE_AIR)) - env.state[trig, 0])
        obs, _, _, done, _ = env.step(a)
    d_pol, d_brk = np.concatenate(d_pol), np.concatenate(d_brk)
    return {'n_steps': int(len(d_pol)), 'nh4_change_policy': float(d_pol.mean()),
            'nh4_change_brake': float(d_brk.mean()), 'brake_higher_pct': float(100 * np.mean(d_brk > d_pol + 1e-12))}


def run_one(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    agent = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    agent.load(Path(a.runs) / f'fcsrl_s{a.seed}' / 'model_final')
    model, pos = fit_predictor(agent, env, all_episodes(splits['train'], dyn.n_members), a.seed)

    def prob(obs):
        return 1 / (1 + np.exp(-np.clip(model.score(features(obs)), -30, 30)))

    act = lambda obs: agent_act(agent, obs)  # noqa: E731
    makers = {'nh4_rule': lambda v: brake(lambda o: np.where(o[:, 0] >= v, BRAKE_AIR, 0.0)),
              'onoff': lambda v: brake(lambda o: np.where(prob(o) >= v, BRAKE_AIR, 0.0)),
              'graded': lambda v: brake(lambda o: BRAKE_AIR * np.minimum(1.0, prob(o) / v))}
    out = {'seed': a.seed, 'fit_positive_rate': pos, 'local_response': local_response(agent, env, splits['val']),
           'none': evaluate(env, splits['val'], act)}
    for kind, values in CURVES.items():
        out[kind] = {str(v): evaluate(env, splits['val'], act, shield=makers[kind](v)) for v in values}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'seed_{a.seed}.json').write_text(json.dumps(out, indent=2))
    print('완료', a.seed, flush=True)


def report(a):
    res = {json.loads(p.read_text())['seed']: json.loads(p.read_text()) for p in Path(a.out).glob('seed_*.json')}
    seeds = sorted(res)
    rs = [res[s] for s in seeds]
    L = [f'FCSRL + PPO-Lag 정책 seed {len(seeds)}개, 평균 ± 표준편차(ddof=0). 검증 40일 x 구성원 5개 = 200 에피소드.', '',
         '## 단계 1: 학습된 정책 (차단 없음, 마지막 정책)', '',
         '| 방법 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 만족 % |', '| --- | --- | --- | --- | --- | --- |']
    n = [r['none'] for r in rs]
    L.append(f"| FCSRL + PPO-Lag (이 실험) | {ms([x['air_m3'] for x in n], 0)} | {ms([x['cost'] for x in n])} | "
             f"{ms([x['nh4_viol_pct'] for x in n])} | {ms([x['do_viol_pct'] for x in n])} | {ms([x['days_within_limit_pct'] for x in n], 1)} |")
    L += ['', '## 단계 2: 사전 점검, 차단 순간의 대리 모델 반응', '',
          '| seed | NH4 >= 3 스텝 | 다음 NH4 변화, 정책 공기량 | 다음 NH4 변화, 5,800 | 5,800 에서 NH4 가 더 높은 비율 |',
          '| --- | --- | --- | --- | --- |']
    for s in seeds:
        lr = res[s]['local_response']
        L.append(f"| {s} | {lr['n_steps']} | {lr['nh4_change_policy']:+.4f} | {lr['nh4_change_brake']:+.4f} | {lr['brake_higher_pct']:.1f}% |")
    L += ['', '## 단계 3: 차단 곡선 (정책 평균)', '',
          '| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 만족 % | 개입 % |',
          '| --- | --- | --- | --- | --- | --- | --- |']

    def row(label, ps):
        return (f"| {label} | {ms([p['air_m3'] for p in ps], 0)} | {ms([p['cost'] for p in ps])} | "
                f"{ms([p['nh4_viol_pct'] for p in ps])} | {ms([p['do_viol_pct'] for p in ps])} | "
                f"{ms([p['days_within_limit_pct'] for p in ps], 1)} | {ms([p.get('intervention_pct', 0) for p in ps], 1)} |")
    L.append(row('차단 없음', n))
    for kind, values in CURVES.items():
        for v in values:
            L.append(row(f'{LABELS[kind]} {v}', [r[kind][str(v)] for r in rs]))
    L += ['', '## 판정용 seed 별 값', '',
          '| seed | H1 theta 3.0 하루 비용 빼기 차단 없음 | H2 켜고 끄기 대 NH4 규칙 (점) | H3 비례 대 켜고 끄기 (점) |',
          '| --- | --- | --- | --- |']
    table = {}
    for s in seeds:
        r = res[s]
        table[('H1', s)] = r['nh4_rule']['3.0']['cost'] - r['none']['cost']
        cells = [f"{table[('H1', s)]:.2f}"]
        for hid, front, ref in (('H2', 'onoff', 'nh4_rule'), ('H3', 'graded', 'onoff')):
            g = gaps(r[front], r[ref])
            table[(hid, s)] = float(np.mean(g)) if g else None
            cells.append(f"{'없음' if not g else f'{np.mean(g):.2f}'} ({len(g)})")
        L.append(f'| {s} | ' + ' | '.join(cells) + ' |')
    L += ['', f'## 사전 등록 판정 (seed {len(seeds)}개 중 3개 이상 음수, 평균 음수)', '']
    for hid, label in (('H1', 'theta 3.0 대 차단 없음'), ('H2', '켜고 끄기 대 NH4 규칙'), ('H3', '비례 대 켜고 끄기')):
        vals = [table[(hid, s)] for s in seeds]
        known = [v for v in vals if v is not None]
        neg = sum(v is not None and v < 0 for v in vals)
        ok = neg >= 3 and bool(known) and np.mean(known) < 0
        mean_txt = f'{np.mean(known):.2f}' if known else '없음'
        L.append(f'- {hid} ({label}): 평균 {mean_txt}, 음수 seed {neg}/{len(seeds)}: {"지지" if ok else "지지 안 됨"}')
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    o.add_argument('--seed', type=int, required=True)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'runs', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
