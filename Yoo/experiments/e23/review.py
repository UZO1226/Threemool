"""Kaggle val 43일 x 모델 5개를 MLP와 GBM으로 평가하고 실제 기록과 구분한다."""
import argparse
import json
from pathlib import Path

import numpy as np
from behavior import make_agent
from env import AerationEnv, AIR_MAX, STEP_H
from evaluate import batched_rollout, episodes
from ppo_config import PPOConfig


def evaluate_final(agent, data, model, gbm, limit=15.):
    result = {}
    for tag, path in [('mlp', model), ('gbm', gbm)]:
        env = AerationEnv(data, path, split='val')
        env.fixed = episodes(env)
        departures = []

        def observe(t, obs, actions, cost, air):
            if actions is not None:
                prior = agent.prior_action(agent._t(agent.obs_normalizer(obs)))[:, 0].cpu().numpy()
                mask = obs[:, 4] == 1
                departures.extend((abs(actions[mask] - prior[mask]) * AIR_MAX / 2).tolist())

        summary, rows = batched_rollout(env, env.sur, lambda o: agent.act_det(o)[:, 0], limit, observe=observe)
        summary['aerobic_prior_departure_nm3h'] = float(np.mean(departures))
        result[tag] = {'summary': summary, 'episodes': rows}
    return result


def references(a):
    agent = make_agent('ppo', 0, a.prepared, PPOConfig(total_steps=9184, n_chunks=1, init_log_std=-1.6))
    result = {'behavior_policy': evaluate_final(agent, a.data, a.model, a.gbm), 'historical_replay': {}}
    for tag, path in [('mlp', a.model), ('gbm', a.gbm)]:
        env = AerationEnv(a.data, path, split='val')
        env.fixed = episodes(env)
        hist = np.stack([np.clip(env.day_data[d]['air'] / AIR_MAX * 2 - 1, -1, 1) for d, _ in env.fixed])
        step = [0]

        def act(obs):
            t = step[0]
            step[0] += 1
            return hist[:, t]

        sm, rows = batched_rollout(env, env.sur, act, 15.)
        result['historical_replay'][tag] = {'summary': sm, 'episodes': rows}
    measured = []
    for d in env.days:
        x = env.day_data[d]
        vn, vd = x['nh4'][1:] > 4, x['do'][1:] > 2
        measured.append({'day': d, 'air_m3': float(x['air'][:-1].sum() * STEP_H),
                         'cost': int(vn.sum() + vd.sum()), 'nh4_viol_pct': float(vn.mean() * 100),
                         'do_viol_pct': float(vd.mean() * 100)})
    result['historical_measured'] = {'days': measured, 'summary': {
        k: float(np.mean([r[k] for r in measured])) for k in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct')}}
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({tag: result['behavior_policy'][tag]['summary'] for tag in ('mlp', 'gbm')}, indent=2))


def report(root, out):
    refs = json.loads((root / 'references.json').read_text())
    rows = []
    for p in sorted((root / 'pilot/runs').glob('*/seed_*/metrics.json')):
        m = json.loads(p.read_text())
        cross = json.loads((p.parent / 'cross_evaluation.json').read_text())
        rows.append({'method': m['method'], 'seed': m['seed'], 'steps': m['steps'],
                     'best': m['val_best'], 'best_feasible': m['best_feasible'],
                     'final': {tag: cross[tag]['summary'] for tag in cross}})
    if len(rows) != 8:
        raise ValueError(f'Expected 8 runs, got {len(rows)}')
    md = ['# e23 Kaggle 복귀 시험 결과', '',
          '학습 및 평가 입력은 Avedore Kaggle 자료다. BSM1은 사용하지 않았다. '
          '모델 안에서의 공기량과 비용이며 실제 공정 절감 또는 안전성 검증이 아니다. '
          '최종 정책은 고정 학습량 뒤 평가했고 test는 사용하지 않았다.', '']
    for tag in ('mlp', 'gbm'):
        md += [f'## {tag.upper()} 평가, val 43일 x 구성원 5개', '',
               '| 방법 | seed | 공기량 Nm3/일 | 하루 비용 (한도 15) | NH4 초과 % | DO 초과 % | 기록 정책과 공기량 차이 Nm3/h |',
               '| --- | --- | --- | --- | --- | --- | --- |']
        for name, r in [('실제 측정', refs['historical_measured']['summary']),
                        ('실제 공기량 재생', refs['historical_replay'][tag]['summary']),
                        ('기록 학습 정책', refs['behavior_policy'][tag]['summary'])]:
            md.append(f"| {name} | | {r['air_m3']:.1f} | {r['cost']:.2f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | |")
        for row in rows:
            r = row['final'][tag]
            md.append(f"| {row['method']} | {row['seed']} | {r['air_m3']:.1f} | {r['cost']:.2f} | "
                      f"{r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {r['aerobic_prior_departure_nm3h']:.1f} |")
        md += ['', '| 방법 | 공기량 평균 ± SD | 비용 평균 ± SD | 한도 만족 seed |', '| --- | --- | --- | --- |']
        for method in ('ppo', 'fcsrl', 'ppo_anchor', 'fcsrl_anchor'):
            rs = [r['final'][tag] for r in rows if r['method'] == method]
            air, cost = [r['air_m3'] for r in rs], [r['cost'] for r in rs]
            md.append(f'| {method} | {np.mean(air):.1f} ± {np.std(air, ddof=1):.1f} | {np.mean(cost):.2f} ± {np.std(cost, ddof=1):.2f} | {sum(c <= 15 for c in cost)}/2 |')
        md.append('')
    baseline = [r for r in rows if r['method'] == 'fcsrl']
    extension = [r for r in rows if r['method'] == 'fcsrl_anchor']
    gate = {}
    for tag in ('mlp', 'gbm'):
        b = [r['final'][tag] for r in baseline]
        e = [r['final'][tag] for r in extension]
        gate[tag] = {'both_seeds_feasible': all(r['cost'] <= 15 for r in e),
                     'mean_air_lower': np.mean([r['air_m3'] for r in e]) < np.mean([r['air_m3'] for r in b]),
                     'mean_cost_not_higher': np.mean([r['cost'] for r in e]) <= np.mean([r['cost'] for r in b])}
    gate = {t: {k: bool(v) for k, v in g.items()} for t, g in gate.items()}
    passed = all(v for g in gate.values() for v in g.values())
    md += ['## 사전 탐색 기준', '', f'FCSRL 행동 정규화 후보 기준 통과: {passed}.', '',
           '```json', json.dumps(gate, indent=2), '```', '',
           'seed 2개 시험이며 통계적 우월성의 확증이 아니다. 모델 간 불일치는 별도로 해석한다. '
           '기록 정책과의 거리 감소는 조건부 행동 분포 안에 있다는 보증이 아니며 대리 모델 편향도 해결하지 않는다.', '']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.md').write_text('\n'.join(md))
    (out / 'summary.json').write_text(json.dumps({'runs': rows, 'gate': gate, 'passed': passed,
        'historical_measured': refs['historical_measured']['summary'],
        'historical_replay': {t: refs['historical_replay'][t]['summary'] for t in ('mlp', 'gbm')},
        'behavior_policy': {t: refs['behavior_policy'][t]['summary'] for t in ('mlp', 'gbm')}}, indent=2))
    print('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('references')
    for name in ('data', 'model', 'gbm', 'prepared', 'out'):
        r.add_argument('--' + name, type=Path, required=True)
    r = sub.add_parser('report')
    r.add_argument('--root', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    references(a) if a.mode == 'references' else report(a.root, a.out)
