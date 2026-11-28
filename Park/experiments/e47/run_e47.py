"""Park/e47 FCSRL 기전 제거 대조 검증 평가 (Park 작성). 규칙은 README.md 를 따른다.
대표와 같은 학습 설정으로 학습한 두 제거 대조 (무작위 고정 표현, feasibility 손실 없음) 의 겹치지 않는 4 x 16 앙상블 묶음 5개에
대표 차단을 붙여 평가한다. FCSRL (대표) 은 Park/e35 orig, PPO-Lag 는 Park/e37 값을 쓴다.

    python run_e47.py run --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --out <결과>/e47
    python run_e47.py report --out <결과>/e47 --runs <결과> --e35 <결과>/e35 --e37 <결과>/e37 --md results.md
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
from run_e38 import BRAKE, COMBOS, ci, key  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

register()
VARIANTS = {'rand': ('fcsrl_rand', 'dw_rand16_runs', 'R'), 'dyn': ('fcsrl_dyn', 'dw_dyn16_runs', 'D')}
AIR_TOL = 0.02   # 대표 평균 공기량이 대조보다 2% 넘게 많으면 비용 판정을 하지 않는다


def load(runs, v, s):
    m, d, p = VARIANTS[v]
    ag = METHODS[m](11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(Path(runs) / d / f'{p}_s{s}' / 'model_final')
    return ag


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    jobs = [(v, c) for v in VARIANTS for c in COMBOS]
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for v, c in jobs[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, v, s): agent_act(g, obs) for s in c])
        r = evaluate(env, splits['val'], ens, shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        out[f'{v}_{key(c)}'] = r
        print(v, c, r, flush=True)
        (Path(a.out) / f'e47_part{a.part}.json').write_text(json.dumps(out, indent=2))


def gather(d, pat):
    res = {}
    for p in sorted(Path(d).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def report(a):
    res = gather(a.out, 'e47_part*.json')
    res.update({k.replace('orig_', 'full_'): v for k, v in gather(a.e35, 'e35_part*.json').items() if k.startswith('orig_')})
    res.update(gather(a.e37, 'e37_part*.json'))   # ppo_<묶음>
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    get = lambda h, k: np.array([res[f'{h}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    names = {'full': 'FCSRL 대표 (표현 학습 전체, Park/e35 orig)', 'dyn': '제거 대조 D: feasibility 손실 없음 (동역학 일관성만)',
             'rand': '제거 대조 R: 무작위 고정 표현 (표현 학습 없음)', 'ppo': '참고: PPO-Lag 같은 설정 (표현 없음, Park/e37)'}
    L = []
    chk = Path(a.runs) / 'dw_chk_runs' / 'A_s0' / 'metrics.json'
    if chk.exists():
        new = json.loads(chk.read_text())['final_val']
        old = json.loads((Path(a.runs) / 'dw_u16_runs' / 'A_s0' / 'metrics.json').read_text())['final_val']
        diff = max(abs(new[k] - old[k]) for k in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct'))
        L += [f"재현 검사: 제거 대조 진입점 (`train_e47.py --method fcsrl`) 으로 다시 학습한 FCSRL seed 0 의 학습 끝 평가가 `dw_u16_runs/A_s0` 기록과 최대 {diff:.2e} 다르다.", '']
    L += ['검증 40일 x 구성원 5개 = 200 에피소드. 모든 조건이 같은 학습 설정 (학습 중 차단 target 3.5, 갱신 16번), 같은 4 x 16 묶음, 같은 대표 차단이다.', '',
          '| 방법 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 차단 개입 % |', '| --- | --- | --- | --- | --- | --- | --- |']
    for h in ('full', 'dyn', 'rand', 'ppo'):
        m = {k: get(h, k).mean() for k in keys}
        L.append(f"| {names[h]} | {m['air_m3']:,.0f} | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} | {m['intervention_pct']:.1f} |")
    L += ['', '## 판정', '']
    for hn, v in (('H1', 'rand'), ('H2', 'dyn')):
        fa, fc, va, vc = get('full', 'air_m3'), get('full', 'cost'), get(v, 'air_m3'), get(v, 'cost')
        ok = fc < vc
        air_ok = fa.mean() <= va.mean() * (1 + AIR_TOL)
        L += [f'### {hn}: 대표 대 {names[v]}', '', '| 묶음 | 대표 공기량 | 대표 비용 | 대조 공기량 | 대조 비용 | 대표 비용이 낮음 |', '| --- | --- | --- | --- | --- | --- |']
        for i, c in enumerate(COMBOS):
            L.append(f"| {key(c)} | {fa[i]:,.0f} | {fc[i]:.3f} | {va[i]:,.0f} | {vc[i]:.3f} | {'예' if ok[i] else '아니오'} |")
        sup = air_ok and ok.sum() >= 4 and fc.mean() < vc.mean()
        verdict = '지지' if sup else ('판정 안 함 (대표 공기량이 2% 넘게 많음)' if not air_ok else '지지 안 됨')
        L += ['', f"- {hn} 판정: 대표 비용이 낮은 묶음 {ok.sum()}/5, 평균 {fc.mean():.3f} 대 {vc.mean():.3f}, 평균 공기량 차이 {100 * (fa.mean() / va.mean() - 1):+.1f}%: {verdict}",
              f"- 참고 비용 차이 (대표 빼기 대조) {ci(fc - vc, '{:+.3f}')}, 공기량 차이 {ci(fa - va, '{:+,.0f}')}", '']
    L += ['## 참고 (판정 없음)', '']
    for x, y in (('dyn', 'rand'), ('rand', 'ppo'), ('full', 'ppo')):
        L.append(f"- {x} 빼기 {y}: 비용 {ci(get(x, 'cost') - get(y, 'cost'), '{:+.3f}')}, 공기량 {ci(get(x, 'air_m3') - get(y, 'air_m3'), '{:+,.0f}')}, DO 위반 {ci(get(x, 'do_viol_pct') - get(y, 'do_viol_pct'), '{:+.2f}')} %p")
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
    for k in ('out', 'runs', 'e35', 'e37', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
