"""Park/e46 NH4 분석계 지연에 대한 견고성 (Park 작성). 규칙은 README.md 를 따른다.
정책과 차단 (ABAC 는 제어기) 이 보는 관측 NH4 를 d 스텝 (15분 x d) 전 값으로 바꾼다. 환경 안의 실제 상태와 비용은 그대로다.

    python run_e46.py run --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --e43 <결과>/e43 --out <결과>/e46
    python run_e46.py report --out <결과>/e46 --e35 <결과>/e35 --e37 <결과>/e37 --e43 <결과>/e43 --md results.md
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
from min_brake import EnsemblePolicy, MinAirBrake  # noqa: E402
from run_e38 import BASE_PPO, BRAKE, COMBOS, ci, key, load  # noqa: E402
from run_e43 import ppo_default  # noqa: E402
from train_darrow import agent_act  # noqa: E402


def jobs():
    slow = [('fcsrl', 1, c) for c in COMBOS] + [('ppo', 1, c) for c in COMBOS] + [('fcsrl', 2, c) for c in COMBOS]
    fast = [('abac', d, None) for d in (1, 2)] + [('ppo_default', 1, s) for s in range(5)]
    return slow, fast


def run(a):
    sel = json.loads((Path(a.e43) / 'pick.json').read_text())
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    slow, fast = jobs()
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for h, d, c in slow[a.part::a.nparts] + fast[a.part::a.nparts]:
        if h in ('fcsrl', 'ppo'):
            ens = EnsemblePolicy([lambda obs, g=load(a.runs, h, s): agent_act(g, obs) for s in c])
            r, _ = rollout_log(env, splits['val'], ens, shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE), delay=d)
            name = f'{h}_d{d}_{key(c)}'
        elif h == 'abac':
            r, _ = rollout_log(env, splits['val'], ABAC(env, *sel['abac']), delay=d)
            name = f'abac_d{d}'
        else:
            r, _ = rollout_log(env, splits['val'], ppo_default(a.runs, c), delay=d)
            name = f'ppo_default_d{d}_{c}'
        out[name] = r
        print(name, r, flush=True)
        (Path(a.out) / f'e46_part{a.part}.json').write_text(json.dumps(out, indent=2))


def gather(d, pat):
    res = {}
    for p in sorted(Path(d).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def report(a):
    sel = json.loads((Path(a.e43) / 'pick.json').read_text())
    res = gather(a.out, 'e46_part*.json')
    res.update({k.replace('orig_', 'fcsrl_d0_'): v for k, v in gather(a.e35, 'e35_part*.json').items() if k.startswith('orig_')})
    res.update({k.replace('ppo_', 'ppo_d0_'): v for k, v in gather(a.e37, 'e37_part*.json').items()})
    v43 = gather(a.e43, 'val_part*.json')
    sp, kp, ki = sel['abac']
    res['abac_d0'] = v43[f'abac|{sp}|{kp}|{ki}']
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')
    get = lambda h, d, k: np.array([res[f'{h}_d{d}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 지연 d = 정책과 차단이 보는 NH4 가 d 스텝 (15분 x d) 전 값. 대표 차단은 모든 RL 조건에 같다.', '',
         '| 방법 | 지연 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % |', '| --- | --- | --- | --- | --- | --- | --- |']
    rowm = lambda n, d, m: f"| {n} | {d} | {m['air_m3']:,.0f} | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} |"  # noqa: E731
    for h, n, ds in (('fcsrl', 'FCSRL 대표 (묶음 5개)', (0, 1, 2)), ('ppo', 'PPO-Lag 같은 설정 + 대표 차단 (Park/e37, 묶음 5개)', (0, 1))):
        for d in ds:
            L.append(rowm(n, f'{d} ({15 * d}분)', {k: get(h, d, k).mean() for k in keys}))
    for d in (0, 1, 2):
        L.append(rowm(f'ABAC (e43 고름: 설정값 {sp}, Kp {kp:g}, Ki {ki:g})', f'{d} ({15 * d}분)', res[f'abac_d{d}']))
    pd1 = {k: np.mean([res[f'ppo_default_d1_{s}'][k] for s in range(5)]) for k in keys}
    L.append(rowm('참고: 기본 PPO-Lag, 차단 없음 (seed 5개)', '1 (15분)', pd1))
    L.append(f"| 기본 PPO-Lag, 차단 없음 (공식 표) | 0 | {BASE_PPO['air_m3']:,.0f} | {BASE_PPO['cost']:.2f} | | | |")
    fa, fc = get('fcsrl', 1, 'air_m3'), get('fcsrl', 1, 'cost')
    ok1 = (fa < BASE_PPO['air_m3']) & (fc < BASE_PPO['cost'])
    dF, dP = fc - get('fcsrl', 0, 'cost'), get('ppo', 1, 'cost') - get('ppo', 0, 'cost')
    ok2 = dF < dP
    L += ['', '## 판정', '', '| 묶음 | 대표 d1 공기량 | 대표 d1 비용 | H1 기준선보다 둘 다 낮음 | 대표 비용 증가 | PPO-Lag 비용 증가 | H2 대표가 덜 늘어남 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {fa[i]:,.0f} | {fc[i]:.3f} | {'예' if ok1[i] else '아니오'} | {dF[i]:+.3f} | {dP[i]:+.3f} | {'예' if ok2[i] else '아니오'} |")
    s1 = ok1.sum() >= 4 and fa.mean() < BASE_PPO['air_m3'] and fc.mean() < BASE_PPO['cost']
    s2 = ok2.sum() >= 4 and dF.mean() < dP.mean()
    L += ['', f"- H1 판정: 기준선보다 둘 다 낮은 묶음 {ok1.sum()}/5, 공기량 {fa.mean():,.0f}, 비용 {fc.mean():.3f}: {'지지' if s1 else '지지 안 됨'}",
          f"- H2 판정: 비용 증가가 PPO-Lag 보다 작은 묶음 {ok2.sum()}/5, 평균 {dF.mean():+.3f} 대 {dP.mean():+.3f}: {'지지' if s2 else '지지 안 됨'}",
          f"- 참고 비용 증가 차이 (대표 빼기 PPO-Lag) {ci(dF - dP, '{:+.3f}')}",
          f"- 참고 대표 지연 1 대 0: 공기량 {ci(fa - get('fcsrl', 0, 'air_m3'), '{:+,.0f}')}, 비용 {ci(dF, '{:+.3f}')}",
          f"- 참고 대표 지연 2 대 0: 공기량 {ci(get('fcsrl', 2, 'air_m3') - get('fcsrl', 0, 'air_m3'), '{:+,.0f}')}, 비용 {ci(get('fcsrl', 2, 'cost') - get('fcsrl', 0, 'cost'), '{:+.3f}')}",
          f"- 참고 ABAC 비용 증가: 지연 1 {res['abac_d1']['cost'] - res['abac_d0']['cost']:+.3f}, 지연 2 {res['abac_d2']['cost'] - res['abac_d0']['cost']:+.3f}"]
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
    for k in ('out', 'e35', 'e37', 'e43', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
