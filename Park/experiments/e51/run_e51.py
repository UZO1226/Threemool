"""Park/e51 새 seed 블록 확인과 차단 개입률 (Park 작성). 규칙은 README.md 를 따른다.
학습은 Park/e47 의 train_e47.py 를 그대로 쓴다 (FCSRL 은 --method fcsrl, 무작위 고정 표현은 --method fcsrl_rand).

    python run_e51.py run --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --out <결과>/e51
    python run_e51.py report --out <결과>/e51 --e43 <결과>/e43 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
from agent import METHODS, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from fcsrl_ablation import register  # noqa: E402
from min_brake import EnsemblePolicy, MinAirBrake  # noqa: E402
from run_e38 import BRAKE, ci, key  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

register()
NEW = [tuple(range(lo, lo + 4)) for lo in (20, 24, 28, 32)]   # 새 seed 블록, 구성원이 겹치지 않는 묶음 4개
VARIANTS = {'full': ('fcsrl', 'dw_s20_runs', 'A'), 'rand': ('fcsrl_rand', 'dw_rand16_runs', 'R')}
OFFICIAL = (90121.0, 10.79)   # 공식 기준선 (기본 PPO-Lag, PROTOCOL v1.1)


def load(runs, v, s):
    m, d, p = VARIANTS[v]
    ag = METHODS[m](11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(Path(runs) / d / f'{p}_s{s}' / 'model_final')
    return ag


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    jobs = [(v, c) for c in NEW for v in VARIANTS]
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for v, c in jobs[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, v, s): agent_act(g, obs) for s in c])
        r = evaluate(env, splits['val'], ens, shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        out[f'{v}_{key(c)}'] = r
        print(v, c, r, flush=True)
        (Path(a.out) / f'e51_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    res = {}
    for p in sorted(Path(a.out).glob('e51_part*.json')):
        res.update(json.loads(p.read_text()))
    val = {}
    for p in sorted(Path(a.e43).glob('val_part*.json')):
        val.update(json.loads(p.read_text()))
    get = lambda h, k: np.array([res[f'{h}_{key(c)}'][k] for c in NEW])  # noqa: E731
    curve = lambda pre: sorted((v['air_m3'], v['cost']) for k, v in val.items() if k.split('|')[0] in (pre, pre + '_curve'))  # noqa: E731
    at = lambda cv, x: float(np.interp(x, [q[0] for q in cv], [q[1] for q in cv], left=np.nan, right=np.nan))  # noqa: E731
    cA, cB = curve('abac'), curve('brake')
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 새 seed 20 ~ 35 로 학습한 정책의 겹치지 않는 4 x 16 묶음 4개, 대표 차단.', '',
         '| 묶음 | FCSRL 공기량 | FCSRL 비용 | FCSRL 개입 % | 무작위 표현 공기량 | 무작위 표현 비용 | 무작위 표현 개입 % |', '| --- | --- | --- | --- | --- | --- | --- |']
    for i, c in enumerate(NEW):
        L.append(f"| {key(c)} | {get('full', 'air_m3')[i]:,.0f} | {get('full', 'cost')[i]:.3f} | {get('full', 'intervention_pct')[i]:.1f} | "
                 f"{get('rand', 'air_m3')[i]:,.0f} | {get('rand', 'cost')[i]:.3f} | {get('rand', 'intervention_pct')[i]:.1f} |")
    L.append(f"| 평균 | {get('full', 'air_m3').mean():,.0f} | {get('full', 'cost').mean():.3f} | {get('full', 'intervention_pct').mean():.1f} | "
             f"{get('rand', 'air_m3').mean():,.0f} | {get('rand', 'cost').mean():.3f} | {get('rand', 'intervention_pct').mean():.1f} |")
    fa, fc, fi, ri = get('full', 'air_m3'), get('full', 'cost'), get('full', 'intervention_pct'), get('rand', 'intervention_pct')
    h1 = (fa < OFFICIAL[0]) & (fc < OFFICIAL[1])
    bA, bB = np.array([at(cA, x) for x in fa]), np.array([at(cB, x) for x in fa])
    h2 = (fc < bA) & (fc < bB)
    h3 = fi < ri
    L += ['', '## 판정', '',
          f"- H1 (새 seed 에서도 공식 기준선보다 둘 다 낮은가): 둘 다 낮은 묶음 {h1.sum()}/4: {'지지' if h1.all() else '지지 안 됨'}",
          f"- H2 (새 seed 에서도 같은 공기량의 ABAC, 차단만보다 비용이 낮은가, Park/e43 곡선을 묶음 공기량에서 보간): "
          + ', '.join(f'{key(c)} {fc[i]:.3f} 대 ABAC {bA[i]:.3f}, 차단만 {bB[i]:.3f}' for i, c in enumerate(NEW))
          + f": 두 기준보다 낮은 묶음 {h2.sum()}/4: {'판정 안 함 (곡선 범위 밖)' if np.isnan(bA).any() or np.isnan(bB).any() else ('지지' if h2.all() else '지지 안 됨')}",
          f"- H3 (FCSRL 의 차단 개입률이 무작위 표현보다 낮은가): 낮은 묶음 {h3.sum()}/4, 평균 {fi.mean():.1f} 대 {ri.mean():.1f}%: {'지지' if h3.all() else '지지 안 됨'}",
          '', '## 참고 (판정 없음)', '',
          f"- FCSRL 빼기 무작위 표현: 비용 {ci(fc - get('rand', 'cost'), '{:+.3f}')}, 공기량 {ci(fa - get('rand', 'air_m3'), '{:+,.0f}')}, 개입률 {ci(fi - ri, '{:+.1f}')} %p",
          f"- 기존 seed 0 ~ 19 의 같은 값 (Park/e35, e47): FCSRL 82,104 / 4.658 / 개입 24.0%, 무작위 표현 80,651 / 4.785 / 45.7%"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e43', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
