"""Park/e25 체크포인트 앙상블 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e25.py run --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과> --out <결과>/e25
    python run_e25.py report --out <결과>/e25 --e19 <결과>/e19 --e23 <결과>/e23 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from agent import FCSRL, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import E15_MARGINS, MinAirBrake  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BRAKE = dict(trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1)   # e19 H1 차단
SEEDS = list(range(10))
KS = {'H1': 3, 'H2': 9}
ENS10 = (81376, 4.80)   # e23 README: 4 x 16 앙상블 조합 10개 평균 (참고)


def run_dir(runs, s):
    return Path(runs) / ('dw_snap_runs' if s < 5 else 'dw_snap10_runs') / f'A_s{s}'


def base_dir(runs, s):
    return Path(runs) / ('dw_minbrake_runs' if s < 5 else 'dw_new66_runs') / f'A_s{s}'


def load(path):
    ag = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(path)
    return ag


def same_weights(p, q):
    a, b = Path(p), Path(q)
    fa = sorted(x.relative_to(a) for x in a.rglob('*') if x.is_file())
    fb = sorted(x.relative_to(b) for x in b.rglob('*') if x.is_file())
    if fa != fb:
        return False
    for f in fa:
        if f.suffix in ('.pt', '.pth'):
            x, y = torch.load(a / f, weights_only=False), torch.load(b / f, weights_only=False)
            xs = x if isinstance(x, dict) else {'v': x}
            ys = y if isinstance(y, dict) else {'v': y}
            for k in xs:
                if torch.is_tensor(xs[k]) and not torch.equal(xs[k], ys[k]):
                    return False
        elif (a / f).read_bytes() != (b / f).read_bytes():
            return False
    return True


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {'same_final': {}, 'H1': {}, 'H2': {}}
    for s in SEEDS:
        d = run_dir(a.runs, s)
        out['same_final'][s] = same_weights(d / 'model_final', base_dir(a.runs, s) / 'model_final')
        for h, k in KS.items():
            ags = [load(d / f'model_u{u}') for u in range(66 - 2 * (k - 1), 66, 2)] + [load(d / 'model_final')]
            act = lambda obs, ags=ags: np.mean([np.asarray(agent_act(g, obs), float).reshape(len(obs)) for g in ags], axis=0)  # noqa: E731
            out[h][s] = evaluate(env, splits['val'], act, shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
            print(h, s, out['same_final'][s], out[h][s], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'e25.json').write_text(json.dumps(out, indent=2))


def report(a):
    res = json.loads((Path(a.out) / 'e25.json').read_text())
    base = {r['seed']: r['conf_l3.5_m0.1_rec'] for r in (json.loads(p.read_text()) for p in Path(a.e19).glob('seed_*.json'))}
    base.update({int(s): r for s, r in json.loads((Path(a.e23) / 'e23.json').read_text())['single'].items()})
    assert sorted(base) == SEEDS
    conds = {'기준: 마지막 정책 하나': base}
    for h, k in KS.items():
        conds[f'{h} k = {k}'] = {int(s): r for s, r in res[h].items()}
    mean = lambda d, key: float(np.mean([d[s][key] for s in SEEDS]))  # noqa: E731
    sd = lambda d, key: float(np.std([d[s][key] for s in SEEDS]))  # noqa: E731
    same = [res['same_final'][str(s)] for s in SEEDS]
    L = ['검증 40일 x 구성원 5개 = 200 에피소드, seed 10개. 모두 e19 H1 차단.', '',
         f"마지막 정책이 기준 정책과 가중치가 같은 seed: {sum(same)}/10", '',
         '| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for label, d in conds.items():
        L.append(f"| {label} (평균 ± 표준편차) | {mean(d, 'air_m3'):.0f} ± {sd(d, 'air_m3'):.0f} | {mean(d, 'cost'):.2f} ± {sd(d, 'cost'):.2f} | "
                 f"{mean(d, 'nh4_viol_pct'):.2f} | {mean(d, 'do_viol_pct'):.2f} | {mean(d, 'days_within_limit_pct'):.1f} | {mean(d, 'intervention_pct'):.1f} |")
    L += [f"| 참고: 4 x 16 앙상블 조합 10개 평균 | {ENS10[0]} | {ENS10[1]:.2f} | | | | |"]
    for h, k in KS.items():
        d = conds[f'{h} k = {k}']
        L += ['', f'## {h} k = {k} seed 별', '', '| seed | 기준 공기량 | 기준 비용 | 공기량 | 비용 | 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- |']
        ok = []
        for s in SEEDS:
            b, n = base[s], d[s]
            ok.append(n['air_m3'] < b['air_m3'] and n['cost'] < b['cost'])
            L.append(f"| {s} | {b['air_m3']:.0f} | {b['cost']:.2f} | {n['air_m3']:.0f} | {n['cost']:.2f} | {'예' if ok[-1] else '아니오'} |")
        ma, mc = mean(d, 'air_m3'), mean(d, 'cost')
        sup = sum(ok) >= 6 and ma < mean(base, 'air_m3') and mc < mean(base, 'cost')
        L += ['', f"- {h} 판정: 둘 다 낮은 seed {sum(ok)}/10, 공기량 {ma:.0f} 대 {mean(base, 'air_m3'):.0f}, 비용 {mc:.2f} 대 {mean(base, 'cost'):.2f}: {'지지' if sup else '지지 안 됨'}",
              f"- 참고: 4 x 16 앙상블 조합 10개 평균보다 둘 다 낮음: {'예' if ma < ENS10[0] and mc < ENS10[1] else '아니오'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e19', 'e23', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
