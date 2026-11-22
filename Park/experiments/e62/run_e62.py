"""Park/e62 NH4 분석계 30분 지연에서 대표와 차단 위 예측 제어의 비교 (Park 작성). 규칙은 README.md 를 따른다.
Park/e46, e53 과 같은 지연 (정책과 차단이 보는 NH4 가 d 스텝 전 값) 에서 차단 위 예측 제어 (Park/e60 고름) 를 보정 없이, 보정 (`DelayComp`) 을 붙여 돈다.
대표 값은 Park/e46 (보정 없음), Park/e53 (보정) 을 쓴다.

    python run_e62.py run --part <i> --nparts <n> --data .. --surrogate .. --e60 <결과>/e60 --out <결과>/e62
    python run_e62.py report --out <결과>/e62 --e60 <결과>/e60 --e46 <결과>/e46 --e53 <결과>/e53 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
sys.path.insert(0, str(HERE.parent / 'e53'))
from audit_brake import rollout_log  # noqa: E402
from darrow_env import BatchEnv, Dynamics, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key  # noqa: E402
from run_e53 import Comped  # noqa: E402

JOBS = [('comp', 2), ('raw', 2), ('comp', 1), ('raw', 1)]   # 30분 (d = 2) 이 주 판정


def gather(d, pat):
    res = {}
    for p in sorted(Path(d).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def run(a):
    sel = json.loads((Path(a.e60) / 'pick.json').read_text())
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for kind, d in JOBS[a.part::a.nparts]:
        act = ShortMPC(env, dyn, margin=sel['margin'], lam=sel['lam'], objective='excess')
        br = MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE)
        if kind == 'comp':
            w = Comped(env, dyn, d, act, br)
            r, _ = rollout_log(env, splits['val'], w.policy, shield=w.brake, delay=d)
        else:
            r, _ = rollout_log(env, splits['val'], act, shield=br, delay=d)
        out[f'mpc_{kind}_d{d}'] = r
        print(kind, d, r, flush=True)
        (Path(a.out) / f'e62_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    m = gather(a.out, 'e62_part*.json')
    m0 = json.loads((Path(a.e60) / 'val.json').read_text())['mpc_brake']
    raw, comp = gather(a.e46, 'e46_part*.json'), gather(a.e53, 'e53_part*.json')
    g = lambda res, d, k: np.array([res[f'fcsrl_d{d}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 지연 d = 정책 (예측 제어) 과 차단이 보는 NH4 가 d 스텝 전 값. 보정 = `DelayComp`.', '',
         '| 방법 | 지연 | 보정 | 공기량 | 하루 비용 |', '| --- | --- | --- | --- | --- |',
         f"| 차단 위 예측 제어 (Park/e60) | 없음 | | {m0['air_m3']:,.0f} | {m0['cost']:.3f} |"]
    for d in (1, 2):
        for kind, res, kn in (('raw', raw, '없음 (대표 Park/e46)'), ('comp', comp, '있음 (대표 Park/e53)')):
            L.append(f"| 대표 (묶음 5개 평균) | {15 * d}분 | {kn} | {g(res, d, 'air_m3').mean():,.0f} | {g(res, d, 'cost').mean():.3f} |")
            r = m[f'mpc_{kind}_d{d}']
            L.append(f"| 차단 위 예측 제어 | {15 * d}분 | {'있음' if kind == 'comp' else '없음'} | {r['air_m3']:,.0f} | {r['cost']:.3f} |")
    rc, mc = g(comp, 2, 'cost'), m['mpc_comp_d2']['cost']
    h1, h2 = rc < mc, rc > mc
    L += ['', '## 판정 (30분 지연, 보정)', '', '| 묶음 | 대표 공기량 | 대표 비용 | 대표가 낮음 |', '| --- | --- | --- | --- |']
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {g(comp, 2, 'air_m3')[i]:,.0f} | {rc[i]:.3f} | {'예' if h1[i] else '아니오'} |")
    L += ['', f"- H1 (대표가 더 낮은가): 낮은 묶음 {h1.sum()}/5, 평균 {rc.mean():.3f} 대 {mc:.3f}: {'지지' if h1.sum() >= 4 and rc.mean() < mc else '지지 안 됨'}",
          f"- H2 (반대 방향): 차단 위 예측 제어가 낮은 묶음 {h2.sum()}/5: {'지지' if h2.sum() >= 4 and rc.mean() > mc else '지지 안 됨'}",
          f"- 참고 비용 차이 (대표 빼기 예측 제어) {ci(rc - mc, '{:+.3f}')}, 공기량 차이 평균 {g(comp, 2, 'air_m3').mean() - m['mpc_comp_d2']['air_m3']:+,.0f}",
          f"- 참고 지연 없음에서 30분 보정으로의 증가: 대표 {rc.mean() - 4.658:+.3f} (Park/e35 4.658 기준), 차단 위 예측 제어 {mc - m0['cost']:+.3f}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'e60', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e60', 'e46', 'e53', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
