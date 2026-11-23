"""Park/e69 학습 정책과 예측 제어의 평균 제어기 (Park 작성). 규칙은 README.md 를 따른다.
대표 묶음마다 매 스텝 학습 정책 공기량과 예측 제어 (초과량, NH4 여유 1.2, 공기 가중치 1.5) 공기량의 평균을 대표 차단에 넣고 검증 구간에서 돈다.
학습 정책이 예측 제어 위에 더하는 몫이 있는지 본다.

    python run_e69.py run --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --out <결과>/e69
    python run_e69.py report --out <결과>/e69 --e35 <결과>/e35 --e58 <결과>/e58 --e59 <결과>/e59 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
sys.path.insert(0, str(HERE.parent / 'e59'))
from darrow_env import BatchEnv, Dynamics, action_to_air, air_to_action, evaluate, load_days  # noqa: E402
from min_brake import EnsemblePolicy, MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key, load  # noqa: E402
from run_e59 import curve_cost, frontier, gather  # noqa: E402
from train_darrow import agent_act  # noqa: E402

MARGIN, LAM = 1.2, 1.5   # Park/e58 의 가중치 1.5 점 (82,238, 대표 평균 공기량과 0.2% 차이)


class Average:
    def __init__(self, policy, mpc):
        self.policy, self.mpc = policy, mpc

    def __call__(self, obs):
        pa = action_to_air(np.asarray(self.policy(obs), dtype=float).reshape(len(obs)))
        return air_to_action(0.5 * (pa + self.mpc.plan(obs)))


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c in COMBOS[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, 'fcsrl', s): agent_act(g, obs) for s in c])
        act = Average(ens, ShortMPC(env, dyn, margin=MARGIN, lam=LAM, objective='excess'))
        r = evaluate(env, splits['val'], act, shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        out[key(c)] = r
        print(c, r, flush=True)
        (Path(a.out) / f'e69_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    v = gather(a.out, 'e69_part*.json')
    e35 = gather(a.e35, 'e35_part*.json')
    mp = gather(a.e58, 'val_part*.json')
    mp.update(gather(a.e59, 'val_part*.json'))
    mf = frontier([(r['air_m3'], r['cost']) for r in mp.values()])
    ks = [key(c) for c in COMBOS]
    ra = np.array([e35[f'orig_{k}']['air_m3'] for k in ks])
    rc = np.array([e35[f'orig_{k}']['cost'] for k in ks])
    va = np.array([v[k]['air_m3'] for k in ks])
    vc = np.array([v[k]['cost'] for k in ks])
    mc = np.array([curve_cost(mf, x) for x in va])
    m15 = mp[str(LAM)]
    L = [f'검증 40일 x 구성원 5개 = 200 에피소드. 평균 제어기 = (학습 정책 공기량 + 예측 제어 공기량) / 2 를 대표 차단에 넣음. 예측 제어만 (공기 가중치 {LAM}, Park/e58) {m15["air_m3"]:,.0f} / {m15["cost"]:.3f}.', '',
         '| 묶음 | 대표 공기량 | 대표 비용 | 평균 제어기 공기량 | 평균 제어기 비용 | 같은 공기량의 예측 제어 곡선 비용 | 차단 개입 % |', '| --- | --- | --- | --- | --- | --- | --- |']
    for i, k in enumerate(ks):
        L.append(f"| {k} | {ra[i]:,.0f} | {rc[i]:.3f} | {va[i]:,.0f} | {vc[i]:.3f} | {'범위 밖' if np.isnan(mc[i]) else f'{mc[i]:.3f}'} | {v[k]['intervention_pct']:.1f} |")
    n1 = int((vc < rc).sum())
    ok2 = np.where(np.isnan(mc), False, vc < np.nan_to_num(mc, nan=-1))
    n2 = int(ok2.sum())
    L += ['', '## 판정', '',
          f"- H1 (평균 제어기가 대표보다 낮은가): 낮은 묶음 {n1}/5, 평균 {vc.mean():.3f} 대 {rc.mean():.3f}: {'지지' if n1 >= 4 and vc.mean() < rc.mean() else '지지 안 됨'}",
          f"- H2 (평균 제어기가 같은 공기량의 예측 제어 곡선보다 낮은가, 학습 정책이 더하는 몫): 낮은 묶음 {n2}/5 (범위 밖 {int(np.isnan(mc).sum())} 개): {'지지' if n2 >= 4 and np.nanmean(vc - mc) < 0 else '지지 안 됨'}",
          f"- 참고 비용 차이 (평균 제어기 빼기 대표) {ci(vc - rc, '{:+.3f}')}, 공기량 차이 평균 {(va - ra).mean():+,.0f}"]
    if not np.isnan(mc).any():
        L.append(f"- 참고 비용 차이 (평균 제어기 빼기 같은 공기량의 곡선) {ci(vc - mc, '{:+.3f}')}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=5)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e35', 'e58', 'e59', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    {'run': run, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
