"""Park/e01 A0, A1 실행 (Park 작성). 규칙은 README.md 의 "사전 고정 조건" 을 따른다.

    export BSM1_ARCHIVE=<BSM1.zip> WANDB_MODE=disabled
    python run_e01.py seed --seed 0 --prepared <결과>/prepared --runs <결과>/runs --out <결과>/e01
    python run_e01.py report --out <결과>/e01 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from agent import FCSRL, Config, ObsNormalizer  # noqa: E402
from env import AerationEnv, STEPS_PER_DAY  # noqa: E402
from evaluate import run  # noqa: E402
from shield import FeasibilityShield, NH4Shield, RandomShield, tau_to_minutes  # noqa: E402

TAUS = (0.7, 0.5, 0.3)      # 큰 값(개입 적음) 부터 확인한다
LIMIT = 72.0
HORIZON = 7                  # A0 정답: t ~ t+6 안에 위반
TRAIN_DAYS = 7
KEYS = ('aeration_kwh_per_day', 'flow_weighted_nh4', 'step_violation_percent',
        'violating_steps_per_day', 'intervention_percent')


def auroc(score, label):
    """Mann-Whitney 방식 AUROC. 동점은 평균 순위로 처리한다."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def day_rollout(agent, state, day, shield=None, scorer=None):
    env = AerationEnv(state, 'dry', start_day=day, days=1)
    obs, _ = env.reset()
    rec = {'nh4_obs': [], 'f': [], 'violation': [], 'used': []}
    while env.t < env.T:
        action = np.asarray(agent.act_det(obs), float).reshape(1)
        if scorer is not None:
            rec['f'].append(scorer.score(obs, action))
        rec['nh4_obs'].append(float(obs[0]))
        used = False
        if shield is not None:
            action, used = shield(obs, action)
        obs, _, _, _, info = env.step(action)
        rec['violation'].append(info['violation'])
        rec['used'].append(float(used))
    return {k: np.array(v) for k, v in rec.items()}


def train_days(agent, states, shield=None, scorer=None):
    return [day_rollout(agent, states[d], d, shield, scorer) for d in range(TRAIN_DAYS)]


def summarize_days(days):
    return {'violating_steps_per_day': float(np.mean([d['violation'].sum() for d in days])),
            'intervention_rate': float(np.mean(np.concatenate([d['used'] for d in days])))}


def a0(days):
    f, n, y = [], [], []
    for d in days:
        T = len(d['violation'])
        for t in range(T - (HORIZON - 1)):
            y.append(d['violation'][t:t + HORIZON].any())
            f.append(d['f'][t])
            n.append(d['nh4_obs'][t])
    return {'n_steps': len(y), 'positive_rate': float(np.mean(y)),
            'auroc_feasibility': auroc(f, y), 'auroc_nh4_obs': auroc(n, y)}


def run_seed(a):
    prepared = Path(a.prepared)
    normalizer = ObsNormalizer(np.load(prepared / 'train_observations.npy'))
    agent = FCSRL(11, 1, Config(), normalizer)
    agent.load(Path(a.runs) / f'fcsrl_s{a.seed}' / 'model_final')
    states = [dict(np.load(prepared / f'day_{d}.npz')) for d in range(TRAIN_DAYS)]
    initial = dict(np.load(prepared / 'initial.npz'))

    scorer = FeasibilityShield(agent, 1.0)
    base = train_days(agent, states, scorer=scorer)
    out = {'seed': a.seed, 'a0': a0(base), 'train_no_shield': summarize_days(base), 'tau_search': {}}

    chosen = None
    for tau in TAUS:
        s = summarize_days(train_days(agent, states, FeasibilityShield(agent, tau)))
        out['tau_search'][str(tau)] = s
        if chosen is None and s['violating_steps_per_day'] <= LIMIT:
            chosen = tau
    out['tau_rule_met'] = chosen is not None
    chosen = chosen if chosen is not None else min(TAUS)
    rate = out['tau_search'][str(chosen)]['intervention_rate']
    nh4 = np.concatenate([d['nh4_obs'] for d in base])
    theta = float(np.quantile(nh4, 1 - rate)) if rate > 0 else float('inf')
    out.update(tau=chosen, tau_minutes=tau_to_minutes(chosen), matched_rate=rate, theta=theta)

    pol = agent.act_det
    shields = {'none': None, 'feasibility': FeasibilityShield(agent, chosen),
               'nh4_rule': NH4Shield(theta), 'random': RandomShield(rate, seed=a.seed)}
    out['eval'] = {}
    for name, sh in shields.items():
        out['eval'][name] = {}
        for w in ('dry', 'rain'):
            if isinstance(sh, RandomShield):
                sh.rng = np.random.default_rng(a.seed)
            week = run(initial, pol, w, shield=sh)['week']
            out['eval'][name][w] = {k: week.get(k, 0.0) for k in KEYS}
        print(a.seed, name, json.dumps(out['eval'][name]['dry']), flush=True)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'seed_{a.seed}.json').write_text(json.dumps(out, indent=2))


def fmt(values, digits):
    v = np.asarray(values, float)
    return f'{v.mean():.{digits}f} ± {v.std():.{digits}f}'


def report(a):
    seeds = [json.loads(p.read_text()) for p in sorted(Path(a.out).glob('seed_*.json'))]
    lines = [f'seed {len(seeds)}개의 평균 ± 표준편차(ddof=0).', '', '## A0 위반 예측 (학습 날짜, 차단 없음)', '',
             '| seed | 양성 비율 | AUROC F(s, a) | AUROC NH4 측정값 |', '| --- | --- | --- | --- |']
    for s in seeds:
        r = s['a0']
        lines.append(f"| {s['seed']} | {r['positive_rate']:.3f} | {r['auroc_feasibility']:.3f} | {r['auroc_nh4_obs']:.3f} |")
    lines += ['', '## tau 선택 (학습 날짜)', '', '| seed | 차단 없음 위반 스텝/일 | tau 0.7 | tau 0.5 | tau 0.3 | 고른 tau | 개입 비율 | theta |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for s in seeds:
        ts = s['tau_search']
        cells = ' | '.join(f"{ts[str(t)]['violating_steps_per_day']:.1f} ({100 * ts[str(t)]['intervention_rate']:.1f}%)" for t in TAUS)
        lines.append(f"| {s['seed']} | {s['train_no_shield']['violating_steps_per_day']:.1f} | {cells} | {s['tau']}"
                     f"{'' if s['tau_rule_met'] else ' (한도 미달)'} | {100 * s['matched_rate']:.1f}% | {s['theta']:.3f} |")
    for w, title in (('dry', '건기'), ('rain', '우기 (분포 밖 점검)')):
        lines += ['', f'## A1 평가, {title}', '', '| 조건 | 폭기 kWh/day | 평균 NH4 | 순간 위반 % | 하루 위반 스텝 | 개입 % |',
                  '| --- | --- | --- | --- | --- | --- |']
        for name, label in (('none', '차단 없음'), ('feasibility', 'F 차단'), ('nh4_rule', '규칙 차단'), ('random', '무작위 차단')):
            vals = [s['eval'][name][w] for s in seeds]
            lines.append(f"| {label} | {fmt([v['aeration_kwh_per_day'] for v in vals], 0)} | "
                         f"{fmt([v['flow_weighted_nh4'] for v in vals], 3)} | {fmt([v['step_violation_percent'] for v in vals], 2)} | "
                         f"{fmt([v['violating_steps_per_day'] for v in vals], 1)} | {fmt([v['intervention_percent'] for v in vals], 1)} |")
    lines += ['', '## seed 별 건기 순간 위반 % (H2 판정용)', '', '| seed | 차단 없음 | F 차단 | 규칙 차단 | 무작위 차단 | F 가 둘보다 낮음 |',
              '| --- | --- | --- | --- | --- | --- |']
    wins = 0
    for s in seeds:
        e = {k: s['eval'][k]['dry']['step_violation_percent'] for k in ('none', 'feasibility', 'nh4_rule', 'random')}
        win = e['feasibility'] < min(e['nh4_rule'], e['random'])
        wins += win
        lines.append(f"| {s['seed']} | {e['none']:.2f} | {e['feasibility']:.2f} | {e['nh4_rule']:.2f} | {e['random']:.2f} | {'예' if win else '아니오'} |")
    mean = {k: np.mean([s['eval'][k]['dry']['step_violation_percent'] for s in seeds]) for k in ('feasibility', 'nh4_rule', 'random')}
    supported = wins >= 2 and mean['feasibility'] < min(mean['nh4_rule'], mean['random'])
    lines += ['', f"H2 판정 규칙 결과: {'지지' if supported else '지지 안 됨'} (이긴 seed {wins}/{len(seeds)})"]
    Path(a.md).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('seed')
    s.add_argument('--seed', type=int, required=True)
    s.add_argument('--prepared', required=True)
    s.add_argument('--runs', required=True)
    s.add_argument('--out', required=True)
    r = sub.add_parser('report')
    r.add_argument('--out', required=True)
    r.add_argument('--md', required=True)
    a = p.parse_args()
    run_seed(a) if a.cmd == 'seed' else report(a)


if __name__ == '__main__':
    main()
