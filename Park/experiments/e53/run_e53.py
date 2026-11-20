"""Park/e53 NH4 분석계 지연 보정 (Park 작성). 규칙은 README.md 를 따른다.
Park/e46 과 같은 지연 (정책과 차단이 보는 NH4 가 d 스텝 전 값) 에서, 정책과 차단 앞에 `DelayComp` 를 붙인다. 보정 없는 값은 Park/e46 을 쓴다.

    python run_e53.py run --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --e43 <결과>/e43 --out <결과>/e53
    python run_e53.py report --out <결과>/e53 --e46 <결과>/e46 --e35 <결과>/e35 --e43 <결과>/e43 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
sys.path.insert(0, str(HERE.parent / 'e43'))
from audit_brake import rollout_log  # noqa: E402
from baselines_darrow import ABAC  # noqa: E402
from darrow_env import BatchEnv, Dynamics, load_days  # noqa: E402
from delay_comp import DelayComp  # noqa: E402
from min_brake import EnsemblePolicy, MinAirBrake  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key, load  # noqa: E402
from train_darrow import agent_act  # noqa: E402

OFFICIAL = (90121.0, 10.79)   # 공식 기준선 (기본 PPO-Lag, PROTOCOL v1.1)


class Comped:
    """정책과 차단이 같은 보정 관측을 보게 한다. act 를 먼저 부르고 같은 스텝의 shield 가 그 관측을 쓴다."""

    def __init__(self, env, dyn, d, act, shield=None):
        self.comp, self.act, self.shield, self.o = DelayComp(env, dyn, d), act, shield, None

    def policy(self, o):
        self.o = self.comp(o)
        return self.act(self.o)

    def brake(self, o, a, env):
        return self.shield(self.o, a, env)


def jobs():
    return [('fcsrl', d, c) for d in (2, 1) for c in COMBOS] + [('abac', d, None) for d in (2, 1)]


def run(a):
    sel = json.loads((Path(a.e43) / 'pick.json').read_text())
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for h, d, c in jobs()[a.part::a.nparts]:
        if h == 'fcsrl':
            ens = EnsemblePolicy([lambda obs, g=load(a.runs, 'fcsrl', s): agent_act(g, obs) for s in c])
            w = Comped(env, dyn, d, ens, MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
            r, _ = rollout_log(env, splits['val'], w.policy, shield=w.brake, delay=d)
            name = f'fcsrl_d{d}_{key(c)}'
        else:
            w = Comped(env, dyn, d, ABAC(env, *sel['abac']))
            r, _ = rollout_log(env, splits['val'], w.policy, delay=d)
            name = f'abac_d{d}'
        out[name] = r
        print(name, r, flush=True)
        (Path(a.out) / f'e53_part{a.part}.json').write_text(json.dumps(out, indent=2))


def gather(d, pat):
    res = {}
    for p in sorted(Path(d).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def report(a):
    sel = json.loads((Path(a.e43) / 'pick.json').read_text())
    comp, raw = gather(a.out, 'e53_part*.json'), gather(a.e46, 'e46_part*.json')
    raw.update({k.replace('orig_', 'fcsrl_d0_'): v for k, v in gather(a.e35, 'e35_part*.json').items() if k.startswith('orig_')})
    sp, kp, ki = sel['abac']
    raw['abac_d0'] = gather(a.e43, 'val_part*.json')[f'abac|{sp}|{kp}|{ki}']
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')
    g = lambda res, d, k: np.array([res[f'fcsrl_d{d}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    row = lambda n, m: f"| {n} | {m['air_m3']:,.0f} | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} |"  # noqa: E731
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 지연 d = 정책과 차단 (ABAC 는 제어기) 이 보는 NH4 가 d 스텝 (15분 x d) 전 값. 보정 = `DelayComp` (Smith 예측기 방식).', '',
         '| 방법 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % |', '| --- | --- | --- | --- | --- | --- |']
    L.append(row('대표, 지연 없음 (Park/e35)', {k: g(raw, 0, k).mean() for k in keys}))
    for d in (1, 2):
        L.append(row(f'대표, {15 * d}분 지연, 보정 없음 (Park/e46)', {k: g(raw, d, k).mean() for k in keys}))
        L.append(row(f'대표, {15 * d}분 지연, 보정', {k: g(comp, d, k).mean() for k in keys}))
    L.append(row('ABAC, 지연 없음 (Park/e43)', raw['abac_d0']))
    for d in (1, 2):
        L.append(row(f'ABAC, {15 * d}분 지연, 보정 없음 (Park/e46)', raw[f'abac_d{d}']))
        L.append(row(f'ABAC, {15 * d}분 지연, 보정', comp[f'abac_d{d}']))
    cc, rc, ca = g(comp, 2, 'cost'), g(raw, 2, 'cost'), g(comp, 2, 'air_m3')
    h1 = cc < rc
    h2 = (ca < OFFICIAL[0]) & (cc < OFFICIAL[1])
    L += ['', '## 판정 (30분 지연)', '', '| 묶음 | 보정 없음 비용 | 보정 비용 | 보정 공기량 | 보정이 낮음 | 기준선보다 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- |']
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {rc[i]:.3f} | {cc[i]:.3f} | {ca[i]:,.0f} | {'예' if h1[i] else '아니오'} | {'예' if h2[i] else '아니오'} |")
    L += ['', f"- H1 (보정이 30분 지연의 비용을 낮추는가): 낮은 묶음 {h1.sum()}/5, 평균 {rc.mean():.3f} 에서 {cc.mean():.3f}: {'지지' if h1.sum() >= 4 and cc.mean() < rc.mean() else '지지 안 됨'}",
          f"- H2 (보정하면 30분 지연에서도 공식 기준선 {OFFICIAL[0]:,.0f} / {OFFICIAL[1]} 보다 둘 다 낮은가): {h2.sum()}/5: {'지지' if h2.all() else '지지 안 됨'}",
          '', '## 참고 (판정 없음)', '',
          f"- 30분 보정 빼기 보정 없음: 비용 {ci(cc - rc, '{:+.3f}')}, 공기량 {ci(ca - g(raw, 2, 'air_m3'), '{:+,.0f}')}",
          f"- 15분 보정 빼기 보정 없음: 비용 {ci(g(comp, 1, 'cost') - g(raw, 1, 'cost'), '{:+.3f}')}",
          f"- 30분 보정 빼기 지연 없음: 비용 {ci(cc - g(raw, 0, 'cost'), '{:+.3f}')}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'runs', 'e43', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e46', 'e35', 'e43', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
