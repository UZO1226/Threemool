"""Park/e03 조기 경보의 학습 표본 밖 확인과 단순 예측기 비교 (Park 작성). 규칙은 README.md 를 따른다.

    export BSM1_ARCHIVE=<BSM1.zip> WANDB_MODE=disabled
    python run_e03.py one --gamma 0.9 --seed 0 --prepared <결과>/prepared --runs <결과>/runs --out <결과>/e03
    python run_e03.py report --out <결과>/e03 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e01'))
sys.path.insert(0, str(HERE.parent / 'e02'))
from agent import FCSRL, Config, ObsNormalizer  # noqa: E402
from env import AerationEnv, STEPS_PER_DAY  # noqa: E402
from shield import FeasibilityShield  # noqa: E402
from run_e01 import auroc  # noqa: E402
from run_e02 import model_dir  # noqa: E402

GAMMAS = (0.9, 0.97, 0.99)
HORIZONS = (12, 24)
SCORE_FROM = 7 * STEPS_PER_DAY
SCORES = ('feasibility', 'nh4_obs', 'time_inflow', 'all_obs')


def harmonics(obs):
    angle = np.arctan2(obs[:, 8], obs[:, 9])
    return np.stack([f(k * angle) for k in (2, 3) for f in (np.sin, np.cos)], axis=1)


def features(obs, kind):
    obs = np.asarray(obs, float)
    if kind == 'time_inflow':
        return np.concatenate([obs[:, [8, 9]], harmonics(obs), obs[:, [5, 6, 7]]], axis=1)
    return np.concatenate([obs, harmonics(obs)], axis=1)


class Logistic:
    """L2 정규화 로지스틱 회귀, 뉴턴법."""

    def fit(self, x, y, l2=1e-3, iters=50):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-8
        z = np.c_[np.ones(len(x)), (x - self.mu) / self.sd]
        w = np.zeros(z.shape[1])
        reg = np.full(len(w), l2)
        reg[0] = 0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
            grad = z.T @ (p - y) / len(y) + reg * w
            hess = (z * (p * (1 - p))[:, None]).T @ z / len(y) + np.diag(reg)
            w -= np.linalg.solve(hess + 1e-9 * np.eye(len(w)), grad)
        self.w = w
        return self

    def score(self, x):
        return np.c_[np.ones(len(x)), (x - self.mu) / self.sd] @ self.w


def continuous_rollout(agent, initial, weather, scorer):
    env = AerationEnv(initial, weather, start_day=0, days=14)
    obs, _ = env.reset()
    rec = {'obs': [], 'f': [], 'violation': []}
    while env.t < env.T:
        action = np.asarray(agent.act_det(obs), float).reshape(1)
        rec['obs'].append(obs.copy())
        rec['f'].append(scorer.score(obs, action))
        obs, _, _, _, info = env.step(action)
        rec['violation'].append(info['violation'])
    return {k: np.array(v) for k, v in rec.items()}


def samples(segments, horizon, start=0):
    """(obs, f, 정답) 을 모은다. 현재 위반이 아니고 t+H 가 구간 안인 스텝만."""
    obs, f, y = [], [], []
    for seg in segments:
        v = seg['violation']
        for t in range(start, len(v) - horizon):
            if v[t]:
                continue
            obs.append(seg['obs'][t])
            f.append(seg['f'][t])
            y.append(v[t + 1:t + horizon + 1].any())
    return np.array(obs), np.array(f), np.array(y, bool)


def run_one(a):
    prepared = Path(a.prepared)
    agent = FCSRL(11, 1, Config(feasibility_discount=a.gamma),
                  ObsNormalizer(np.load(prepared / 'train_observations.npy')))
    agent.load(model_dir(a.runs, a.gamma, a.seed))
    states = [dict(np.load(prepared / f'day_{d}.npz')) for d in range(7)]
    initial = dict(np.load(prepared / 'initial.npz'))
    scorer = FeasibilityShield(agent, 1.0)

    # 학습 날짜. 로지스틱 회귀 입력에 관측 전체가 필요하므로 obs 를 함께 기록한다 (e02 와 같은 궤적)
    train = []
    for day, s in enumerate(states):
        env = AerationEnv(s, 'dry', start_day=day, days=1)
        obs, _ = env.reset()
        rec = {'obs': [], 'f': [], 'violation': []}
        while env.t < env.T:
            action = np.asarray(agent.act_det(obs), float).reshape(1)
            rec['obs'].append(obs.copy())
            rec['f'].append(scorer.score(obs, action))
            obs, _, _, _, info = env.step(action)
            rec['violation'].append(info['violation'])
        train.append({k: np.array(v) for k, v in rec.items()})
    evals = {w: continuous_rollout(agent, initial, w, scorer) for w in ('dry', 'rain')}

    out = {'gamma': a.gamma, 'seed': a.seed, 'results': {}}
    for h in HORIZONS:
        tr_obs, _, tr_y = samples(train, h)
        models = {k: Logistic().fit(features(tr_obs, k), tr_y.astype(float)) for k in ('time_inflow', 'all_obs')}
        out['results'][str(h)] = {}
        for split, segs, start in (('train', train, 0), ('eval_dry', [evals['dry']], SCORE_FROM),
                                   ('eval_rain', [evals['rain']], SCORE_FROM)):
            obs, f, y = samples(segs, h, start)
            scores = {'feasibility': f, 'nh4_obs': obs[:, 0],
                      'time_inflow': models['time_inflow'].score(features(obs, 'time_inflow')),
                      'all_obs': models['all_obs'].score(features(obs, 'all_obs'))}
            out['results'][str(h)][split] = {'n': int(len(y)), 'positive_rate': float(y.mean()),
                                             **{k: auroc(v, y) for k, v in scores.items()}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'g{a.gamma}_s{a.seed}.json').write_text(json.dumps(out, indent=2))
    print('완료', a.gamma, a.seed, flush=True)


def ms(v):
    v = np.asarray(v, float)
    return f'{v.mean():.3f} ± {v.std():.3f}'


def report(a):
    res = {}
    for p in Path(a.out).glob('g*_s*.json'):
        r = json.loads(p.read_text())
        res[(r['gamma'], r['seed'])] = r['results']
    seeds = sorted({s for _, s in res})
    L = [f'seed {len(seeds)}개, AUROC 평균 ± 표준편차(ddof=0). 현재 위반이 아닌 스텝만.', '']
    names = {'feasibility': 'F', 'nh4_obs': 'NH4 측정값', 'time_inflow': '시각과 유입', 'all_obs': '관측 전체'}
    splits = {'train': '학습 날짜 (표본 안)', 'eval_dry': '평가 건기 (주 판정)', 'eval_rain': '평가 우기'}
    for h in HORIZONS:
        L += [f'## H = {h} ({h * 5}분)', '', '| 자료 | gamma_F | 양성 비율 | ' + ' | '.join(names.values()) + ' |',
              '| --- | --- | --- | ' + ' | '.join('---' for _ in names) + ' |']
        for split, label in splits.items():
            for g in GAMMAS:
                rs = [res[(g, s)][str(h)][split] for s in seeds if (g, s) in res]
                L.append(f"| {label} | {g} | {ms([r['positive_rate'] for r in rs])} | "
                         + ' | '.join(ms([r[k] for r in rs]) for k in names) + ' |')
        L.append('')
    L += ['## seed 별 평가 건기 H = 12', '', '| gamma_F | seed | ' + ' | '.join(names.values()) + ' |',
          '| --- | --- | ' + ' | '.join('---' for _ in names) + ' |']
    for g in GAMMAS:
        for s in seeds:
            if (g, s) in res:
                r = res[(g, s)]['12']['eval_dry']
                L.append(f'| {g} | {s} | ' + ' | '.join(f'{r[k]:.3f}' for k in names) + ' |')
    main = [res[(0.9, s)]['12']['eval_dry'] for s in seeds if (0.9, s) in res]
    L += ['', '## 사전 등록 판정 (gamma_F 0.9, 평가 건기, H = 12)', '']
    for hid, other in (('H1', 'nh4_obs'), ('H2', 'time_inflow'), ('H3', 'all_obs')):
        f = np.array([r['feasibility'] for r in main])
        o = np.array([r[other] for r in main])
        ok = (f > o).sum() >= 2 and f.mean() > o.mean()
        L.append(f"- {hid} F {f.mean():.3f} 대 {names[other]} {o.mean():.3f}, F 가 높은 seed {(f > o).sum()}/{len(f)}: "
                 f"{'지지' if ok else '지지 안 됨'}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    o.add_argument('--gamma', type=float, required=True)
    o.add_argument('--seed', type=int, required=True)
    o.add_argument('--prepared', required=True)
    o.add_argument('--runs', required=True)
    o.add_argument('--out', required=True)
    r = sub.add_parser('report')
    r.add_argument('--out', required=True)
    r.add_argument('--md', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
