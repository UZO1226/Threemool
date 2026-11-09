"""Park/e06 우기에서 차단 순서가 유지되는가 (Park 작성). 규칙은 README.md 를 따른다.

e05 와 같은 곡선을 우기 평가로 돌린다. 예측기는 건기 학습 날짜로 맞춘다.

    export BSM1_ARCHIVE=<BSM1.zip> WANDB_MODE=disabled
    python run_e06.py one --gamma 0.9 --seed 0 --prepared <결과>/prepared --runs <결과>/runs --out <결과>/e06
    python run_e06.py report --out <결과>/e06 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
for name in ('e01', 'e02', 'e03', 'e04', 'e05'):
    sys.path.insert(0, str(HERE.parent / name))
from agent import FCSRL, Config, ObsNormalizer  # noqa: E402
from evaluate import run  # noqa: E402
from shield import GradedShield, NH4Shield, PredictorShield  # noqa: E402
from run_e02 import model_dir  # noqa: E402
from run_e03 import Logistic, features  # noqa: E402
from run_e04 import train_samples  # noqa: E402
import run_e05  # noqa: E402

WEATHER = 'rain'


def point(w):
    return {'kwh': w['aeration_kwh_per_day'], 'violation': w['step_violation_percent'],
            'intervention': w.get('intervention_percent', 0.0)}


def run_one(a):
    prepared = Path(a.prepared)
    agent = FCSRL(11, 1, Config(feasibility_discount=a.gamma),
                  ObsNormalizer(np.load(prepared / 'train_observations.npy')))
    agent.load(model_dir(a.runs, a.gamma, a.seed))
    states = [dict(np.load(prepared / f'day_{d}.npz')) for d in range(7)]
    initial = dict(np.load(prepared / 'initial.npz'))
    x, y = train_samples(agent, states)          # 건기 학습 날짜
    model = Logistic().fit(features(x, 'all_obs'), y)

    def prob(obs):
        z = model.score(features(np.atleast_2d(obs), 'all_obs'))[0]
        return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))

    makers = {'graded': lambda v: GradedShield(prob, v), 'onoff': lambda v: PredictorShield(prob, v),
              'nh4_rule': lambda v: NH4Shield(v)}
    out = {'gamma': a.gamma, 'seed': a.seed, 'weather': WEATHER,
           'none': point(run(initial, agent.act_det, WEATHER)['week'])}
    for kind, values in run_e05.CURVES.items():
        out[kind] = {str(v): point(run(initial, agent.act_det, WEATHER, shield=makers[kind](v))['week']) for v in values}
        print(a.gamma, a.seed, kind, 'done', flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'g{a.gamma}_s{a.seed}.json').write_text(json.dumps(out, indent=2))


def report(a):
    run_e05.report(a)
    md = Path(a.md)
    md.write_text(md.read_text().replace('건기 평가 구간', '우기 평가 구간'))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    for k in ('prepared', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    o.add_argument('--gamma', type=float, required=True)
    o.add_argument('--seed', type=int, required=True)
    r = sub.add_parser('report')
    r.add_argument('--out', required=True)
    r.add_argument('--md', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
