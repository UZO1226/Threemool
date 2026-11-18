"""Park/e39 차단 정보 제한 검증 평가 (Park 작성). 규칙은 README.md 를 따른다.
환경은 그대로 GBM 대리 모델이고, 차단 예측의 입력 중 측정할 수 없는 유입 변수 4개 (inf_s_nd, inf_x_nd, inf_s_s, inf_x_s) 를
학습 구간 평균으로 바꾼다. 정책은 FCSRL 대표 (e22 4 x 16 앙상블) 와 같은 설정의 PPO-Lag (e37), 구성원이 겹치지 않는 묶음 5개.
정보를 모두 아는 값은 Park/e35 orig (FCSRL) 와 Park/e37 (PPO-Lag) 를 쓴다.

    python run_e39.py run --part <i> --nparts <n> --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과> --out <결과>/e39
    python run_e39.py report --out <결과>/e39 --e35 <결과>/e35 --e37 <결과>/e37 --md results.md
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
from darrow_env import EXO, STATE, BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import E15_MARGINS, EnsemblePolicy, MinAirBrake  # noqa: E402
from mlp_surrogate import MaskedSurrogate  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BRAKE = dict(trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1, plan=(4, 1.8, 500.0))   # 대표 차단 (원판), 여유는 다시 맞추지 않는다
COMBOS = [tuple(range(lo, lo + 4)) for lo in (0, 4, 8, 12, 16)]   # 구성원이 겹치지 않는 묶음 5개
key = lambda c: '_'.join(map(str, c))  # noqa: E731
HIDDEN = ('inf_s_nd', 'inf_x_nd', 'inf_s_s', 'inf_x_s')   # 정책이 못 보고 처리장에서 실시간으로 재지 않는 유입 변수
COND = [(h, c) for h in ('fcsrl', 'ppo') for c in COMBOS]
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
    env = BatchEnv(day_data, dyn)
    idx = [EXO.index(k) for k in HIDDEN]
    mean = np.vstack([day_data[d]['exo'] for d in splits['train']]).mean(0)[idx]   # 학습 구간 평균
    masked = MaskedSurrogate(dyn, [len(STATE) + i for i in idx], mean)
    out = {}
    for h, c in COND[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, h, s): agent_act(g, obs) for s in c])
        br = MinAirBrake(masked, 3.5, 1, 'others', **BRAKE)
        r = evaluate(env, splits['val'], ens, shield=br)
        r['plan_steps'] = br.plan_steps
        out[f'{h}_mask_{key(c)}'] = r
        print(h, c, r, flush=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f'e39_part{a.part}.json').write_text(json.dumps(out, indent=2))


def ci(diff, f):
    n = len(diff)
    hw = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    return f"{f.format(diff.mean())} (95% 신뢰구간 {f.format(diff.mean() - hw)} ~ {f.format(diff.mean() + hw)})"


def report(a):
    res = {}
    for p in sorted(Path(a.out).glob('e39_part*.json')):
        res.update(json.loads(p.read_text()))
    for p in sorted(Path(a.e35).glob('e35_part*.json')):
        res.update({k.replace('orig_', 'fcsrl_full_'): v for k, v in json.loads(p.read_text()).items() if k.startswith('orig_')})
    for p in sorted(Path(a.e37).glob('e37_part*.json')):
        res.update({k.replace('ppo_', 'ppo_full_'): v for k, v in json.loads(p.read_text()).items()})
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    get = lambda h, w, k: np.array([res[f'{h}_{w}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    name = {'fcsrl': 'FCSRL 대표', 'ppo': 'PPO-Lag 같은 설정'}
    wl = {'full': '모두 앎 (지금, e35, e37 값)', 'mask': '유입 변수 4개 모름'}
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 환경은 GBM, 차단 예측 입력의 유입 변수 4개 (' + ', '.join(HIDDEN) + ') 를 학습 구간 평균으로 바꿨다. '
         '구성원이 겹치지 않는 4 x 16 앙상블 묶음 5개의 평균.', '',
         '| 정책 | 차단 정보 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for h in ('fcsrl', 'ppo'):
        for w in ('full', 'mask'):
            m = {k: get(h, w, k).mean() for k in keys}
            L.append(f"| {name[h]} | {wl[w]} | {m['air_m3']:.0f} | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} "
                     f"| {m['days_within_limit_pct']:.1f} | {m['intervention_pct']:.1f} |")
    L.append(f"| 기본 PPO-Lag, 차단 없음 (RESULT.md 공식 표) | | {BASE_PPO['air_m3']:.0f} | {BASE_PPO['cost']:.2f} | | | | |")
    L += ['', '## H1 정보 제한 차단의 FCSRL 대표 대 기본 PPO-Lag (차단 없음)', '',
          '| 묶음 | 공기량 | 비용 | NH4 위반 % | DO 위반 % | 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- |']
    ok = []
    for c in COMBOS:
        r = res[f'fcsrl_mask_{key(c)}']
        ok.append(r['air_m3'] < BASE_PPO['air_m3'] and r['cost'] < BASE_PPO['cost'])
        L.append(f"| {key(c)} | {r['air_m3']:.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {'예' if ok[-1] else '아니오'} |")
    fa, fc = get('fcsrl', 'mask', 'air_m3').mean(), get('fcsrl', 'mask', 'cost').mean()
    sup = sum(ok) >= 4 and fa < BASE_PPO['air_m3'] and fc < BASE_PPO['cost']
    L += ['', f"- H1 판정: 둘 다 낮은 묶음 {sum(ok)}/5, 공기량 {fa:.0f} 대 {BASE_PPO['air_m3']:.0f}, 비용 {fc:.3f} 대 {BASE_PPO['cost']:.2f}: {'지지' if sup else '지지 안 됨'}"]
    L += ['', '## H2 비용 증가 (정보 제한 빼기 모두 앎): FCSRL 대 PPO-Lag', '',
          '| 묶음 | FCSRL 비용 증가 | PPO-Lag 비용 증가 | FCSRL 공기량 변화 | PPO-Lag 공기량 변화 | FCSRL 이 덜 늘어남 |', '| --- | --- | --- | --- | --- | --- |']
    dfc, dpc = get('fcsrl', 'mask', 'cost') - get('fcsrl', 'full', 'cost'), get('ppo', 'mask', 'cost') - get('ppo', 'full', 'cost')
    dfa, dpa = get('fcsrl', 'mask', 'air_m3') - get('fcsrl', 'full', 'air_m3'), get('ppo', 'mask', 'air_m3') - get('ppo', 'full', 'air_m3')
    ok2 = dfc < dpc
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {dfc[i]:+.3f} | {dpc[i]:+.3f} | {dfa[i]:+,.0f} | {dpa[i]:+,.0f} | {'예' if ok2[i] else '아니오'} |")
    sup2 = ok2.sum() >= 4 and dfc.mean() < dpc.mean()
    L += ['', f"- H2 판정: FCSRL 비용 증가가 더 작은 묶음 {ok2.sum()}/5, 평균 {dfc.mean():+.3f} 대 {dpc.mean():+.3f}: {'지지' if sup2 else '지지 안 됨'}",
          f"- 참고 비용 증가 차이 (FCSRL 빼기 PPO-Lag) {ci(dfc - dpc, '{:+.3f}')}",
          f"- 참고 공기량 변화 차이 (FCSRL 빼기 PPO-Lag) {ci(dfa - dpa, '{:+,.0f}')}"]
    for h in ('fcsrl', 'ppo'):
        L.append(f"- 참고 {name[h]} 정보 제한 대 모두 앎: 공기량 {ci(get(h, 'mask', 'air_m3') - get(h, 'full', 'air_m3'), '{:+,.0f}')}, "
                 f"비용 {ci(get(h, 'mask', 'cost') - get(h, 'full', 'cost'), '{:+.3f}')}, NH4 위반 {ci(get(h, 'mask', 'nh4_viol_pct') - get(h, 'full', 'nh4_viol_pct'), '{:+.2f}')}")
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
    for k in ('out', 'e35', 'e37', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
