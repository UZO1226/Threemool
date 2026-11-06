"""저장된 마지막 정책의 학습 및 검증 분포 비용을 사후 진단한다.

결정적 평균 행동으로 재평가하며 학습 중 확률적 수집 CVaR와 구분한다.
정책 선택과 하이퍼파라미터 변경에 쓰지 않는다.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/wwtp'))
import torch
from agents import load_agent
from env import AerationEnv
from evaluate import batched_rollout, episodes, summarize_rows
from ppo_tail import risk_metrics


def decompose(report):
    """초기 NH4 두 집단의 구성비와 집단 내부 비용 차이를 기술적으로 분해한다."""
    results = []
    for row in report:
        train, val = row['train'], row['val']
        tg, vg = train['by_initial_nh4'], val['by_initial_nh4']
        n = sum(v['episodes'] for v in vg.values())
        reweighted = sum(vg[k]['episodes'] / n * tg[k]['cost'] for k in tg)
        composition = reweighted - train['metrics']['cost']
        within = val['metrics']['cost'] - reweighted
        assert abs(composition + within - (val['metrics']['cost'] - train['metrics']['cost'])) < 1e-10
        results.append({'variant': row['variant'], 'seed': row['seed'],
            'train_cost': train['metrics']['cost'], 'val_cost': val['metrics']['cost'],
            'train_cost_reweighted_to_val_initial_nh4_mix': reweighted,
            'composition_term': composition, 'within_group_term': within})
    return {'note': 'Descriptive decomposition, not a causal identification or a new policy evaluation.',
            'results': results}


def review(root, output, data, surrogate):
    torch.set_num_threads(1)
    assert json.loads((root / 'status.json').read_text())['phase'] == 'complete'
    envs = {s: AerationEnv(data, surrogate, split=s) for s in ('train', 'val')}
    for env in envs.values():
        env.fixed = episodes(env)
    results = json.loads((root / 'results.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for result in results:
        method, seed = result['variant'], result['seed']
        run = root / 'runs' / method / f'seed_{seed}'
        agent = load_agent(run, envs['train'], cudaid=-1, checkpoint='final')
        row = {'variant': method, 'seed': seed, 'checkpoint': 'final', 'action': 'deterministic_mean'}
        for split, env in envs.items():
            metrics, rows = batched_rollout(env, env.sur, lambda obs: agent.act_det(obs)[:, 0], 15)
            metrics.update(risk_metrics(rows))
            if split == 'val':
                for key in ('cost', 'cost_cvar', 'air_m3', 'reward'):
                    assert abs(metrics[key] - result['val_final'][key]) < 1e-10, (method, seed, key)
            member_stats = {str(m): summarize_rows([r for r in rows if r['member'] == m], 15)
                            for m in range(env.n_members)}
            groups = {}
            for high in (False, True):
                group_rows = [r for r in rows if (env.day_data[r['day']]['nh4'][0] > 4) == high]
                groups['initial_nh4_gt4' if high else 'initial_nh4_le4'] = {
                    'episodes': len(group_rows), 'cost': statistics.mean(r['cost'] for r in group_rows),
                    'cost_cvar': risk_metrics(group_rows)['cost_cvar']}
            row[split] = {'metrics': metrics, 'by_member': member_stats, 'by_initial_nh4': groups}
            (output / f'{method}_s{seed}_{split}_episodes.json').write_text(json.dumps(rows, indent=2))
        report.append(row)
        print(method, seed, 'train CVaR', row['train']['metrics']['cost_cvar'],
              'val CVaR', row['val']['metrics']['cost_cvar'], flush=True)
    (output / 'train_val_final.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / 'initial_mix_decomposition.json').write_text(json.dumps(decompose(report), indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    a = parser.parse_args()
    review(a.runs, a.out, a.data, a.model)
