"""Park/e38 차단 모델 불일치 검증 평가 (Park 작성). 규칙은 README.md 를 따른다.
환경은 그대로 GBM 대리 모델이고, 차단 예측만 (1 - w) x GBM + w x MLP 로 바꾼다 (w = 0.5 주 판정, w = 1.0 참고).
정책은 FCSRL 대표 (e22 4 x 16 앙상블) 와 같은 설정의 PPO-Lag (e37), 구성원이 겹치지 않는 묶음 5개.
w = 0 값은 Park/e35 orig (FCSRL) 와 Park/e37 (PPO-Lag) 를 쓴다.

    python run_e38.py run --part <i> --nparts <n> --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --mlp <darrow_surrogate_mlp.npz> --runs <결과> --out <결과>/e38
    python run_e38.py report --out <결과>/e38 --e35 <결과>/e35 --e37 <결과>/e37 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from agent import METHODS, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import E15_MARGINS, EnsemblePolicy, MinAirBrake  # noqa: E402
from mlp_surrogate import MixedSurrogate, MLPSurrogate  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BRAKE = dict(trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1, plan=(4, 1.8, 500.0))   # 대표 차단 (원판), 여유는 다시 맞추지 않는다
COMBOS = [tuple(range(lo, lo + 4)) for lo in (0, 4, 8, 12, 16)]   # 구성원이 겹치지 않는 묶음 5개
key = lambda c: '_'.join(map(str, c))  # noqa: E731
WS = {'w05': 0.5, 'w10': 1.0}
COND = [(w, h, c) for w in WS for h in ('fcsrl', 'ppo') for c in COMBOS]   # w = 0.5 를 먼저 돈다
BASE_PPO = {'air_m3': 90121.0, 'cost': 10.79}   # RESULT.md 공식 표 기본 PPO-Lag (차단 없음) seed 5개 평균


def path(runs, h, s):
    if h == 'ppo':
        return Path(runs) / 'dw_ppo16_runs' / f'P_s{s}' / 'model_final'
    sub = 'dw_u16_runs' if s < 5 else ('dw_new16_runs' if s < 10 else 'dw_rep16_runs')
    return Path(runs) / sub / f'A_s{s}' / 'model_final'


def load(runs, h, s):
    ag = METHODS[h](11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(path(runs, h, s))
    return ag


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    mlp = MLPSurrogate(a.mlp)
    env = BatchEnv(day_data, dyn)
    out = {}
    for w, h, c in COND[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, h, s): agent_act(g, obs) for s in c])
        br = MinAirBrake(MixedSurrogate(dyn, mlp, WS[w]), 3.5, 1, 'others', **BRAKE)
        r = evaluate(env, splits['val'], ens, shield=br)
        r['plan_steps'] = br.plan_steps
        out[f'{h}_{w}_{key(c)}'] = r
        print(h, w, c, r, flush=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f'e38_part{a.part}.json').write_text(json.dumps(out, indent=2))


def ci(diff, f):
    n = len(diff)
    hw = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    return f"{f.format(diff.mean())} (95% 신뢰구간 {f.format(diff.mean() - hw)} ~ {f.format(diff.mean() + hw)})"


def report(a):
    res = {}
    for p in sorted(Path(a.out).glob('e38_part*.json')):
        res.update(json.loads(p.read_text()))
    for p in sorted(Path(a.e35).glob('e35_part*.json')):
        res.update({k.replace('orig_', 'fcsrl_w00_'): v for k, v in json.loads(p.read_text()).items() if k.startswith('orig_')})
    for p in sorted(Path(a.e37).glob('e37_part*.json')):
        res.update({k.replace('ppo_', 'ppo_w00_'): v for k, v in json.loads(p.read_text()).items()})
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    get = lambda h, w, k: np.array([res[f'{h}_{w}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    name = {'fcsrl': 'FCSRL 대표', 'ppo': 'PPO-Lag 같은 설정'}
    wl = {'w00': 'w = 0 (지금, e35, e37 값)', 'w05': 'w = 0.5', 'w10': 'w = 1.0 (MLP)'}
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 환경은 GBM, 차단 예측은 (1 - w) x GBM + w x MLP (다른 구성원 4개 평균). '
         '구성원이 겹치지 않는 4 x 16 앙상블 묶음 5개의 평균.', '',
         '| 정책 | 차단 모델 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for h in ('fcsrl', 'ppo'):
        for w in ('w00', 'w05', 'w10'):
            m = {k: get(h, w, k).mean() for k in keys}
            L.append(f"| {name[h]} | {wl[w]} | {m['air_m3']:.0f} | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} "
                     f"| {m['days_within_limit_pct']:.1f} | {m['intervention_pct']:.1f} |")
    L.append(f"| 기본 PPO-Lag, 차단 없음 (RESULT.md 공식 표) | | {BASE_PPO['air_m3']:.0f} | {BASE_PPO['cost']:.2f} | | | | |")
    # H1
    L += ['', '## H1 w = 0.5 의 FCSRL 대표 대 기본 PPO-Lag (차단 없음)', '',
          '| 묶음 | 공기량 | 비용 | NH4 위반 % | DO 위반 % | 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- |']
    ok = []
    for c in COMBOS:
        r = res[f'fcsrl_w05_{key(c)}']
        ok.append(r['air_m3'] < BASE_PPO['air_m3'] and r['cost'] < BASE_PPO['cost'])
        L.append(f"| {key(c)} | {r['air_m3']:.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {'예' if ok[-1] else '아니오'} |")
    fa, fc = get('fcsrl', 'w05', 'air_m3').mean(), get('fcsrl', 'w05', 'cost').mean()
    sup = sum(ok) >= 4 and fa < BASE_PPO['air_m3'] and fc < BASE_PPO['cost']
    L += ['', f"- H1 판정: 둘 다 낮은 묶음 {sum(ok)}/5, 공기량 {fa:.0f} 대 {BASE_PPO['air_m3']:.0f}, 비용 {fc:.3f} 대 {BASE_PPO['cost']:.2f}: {'지지' if sup else '지지 안 됨'}"]
    ok10 = [res[f'fcsrl_w10_{key(c)}']['air_m3'] < BASE_PPO['air_m3'] and res[f'fcsrl_w10_{key(c)}']['cost'] < BASE_PPO['cost'] for c in COMBOS]
    L.append(f"- 참고 같은 비교를 w = 1.0 에서: 둘 다 낮은 묶음 {sum(ok10)}/5")
    # H2
    L += ['', '## H2 비용 증가 (w = 0.5 빼기 w = 0): FCSRL 대 PPO-Lag', '',
          '| 묶음 | FCSRL 비용 증가 | PPO-Lag 비용 증가 | FCSRL 공기량 변화 | PPO-Lag 공기량 변화 | FCSRL 이 덜 늘어남 |', '| --- | --- | --- | --- | --- | --- |']
    dfc, dpc = get('fcsrl', 'w05', 'cost') - get('fcsrl', 'w00', 'cost'), get('ppo', 'w05', 'cost') - get('ppo', 'w00', 'cost')
    dfa, dpa = get('fcsrl', 'w05', 'air_m3') - get('fcsrl', 'w00', 'air_m3'), get('ppo', 'w05', 'air_m3') - get('ppo', 'w00', 'air_m3')
    ok2 = dfc < dpc
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {dfc[i]:+.3f} | {dpc[i]:+.3f} | {dfa[i]:+,.0f} | {dpa[i]:+,.0f} | {'예' if ok2[i] else '아니오'} |")
    sup2 = ok2.sum() >= 4 and dfc.mean() < dpc.mean()
    L += ['', f"- H2 판정: FCSRL 비용 증가가 더 작은 묶음 {ok2.sum()}/5, 평균 {dfc.mean():+.3f} 대 {dpc.mean():+.3f}: {'지지' if sup2 else '지지 안 됨'}",
          f"- 참고 비용 증가 차이 (FCSRL 빼기 PPO-Lag) {ci(dfc - dpc, '{:+.3f}')}",
          f"- 참고 공기량 변화 차이 (FCSRL 빼기 PPO-Lag) {ci(dfa - dpa, '{:+,.0f}')}"]
    d10 = (get('fcsrl', 'w10', 'cost') - get('fcsrl', 'w00', 'cost')) - (get('ppo', 'w10', 'cost') - get('ppo', 'w00', 'cost'))
    L.append(f"- 참고 같은 비교를 w = 1.0 에서: FCSRL 비용 증가가 더 작은 묶음 {(d10 < 0).sum()}/5, 차이 {ci(d10, '{:+.3f}')}")
    for h in ('fcsrl', 'ppo'):
        for w in ('w05', 'w10'):
            L.append(f"- 참고 {name[h]} {wl[w]} 대 w = 0: 공기량 {ci(get(h, w, 'air_m3') - get(h, 'w00', 'air_m3'), '{:+,.0f}')}, "
                     f"비용 {ci(get(h, w, 'cost') - get(h, 'w00', 'cost'), '{:+.3f}')}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'mlp', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e35', 'e37', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
