"""Park/e61 차단 모델이 틀릴 때 대표와 차단 위 예측 제어의 비교 (Park 작성). 규칙은 README.md 를 따른다.
Park/e38 과 같이 환경은 GBM 대리 모델 그대로 두고, 차단과 예측 제어가 쓰는 예측만 (1 - w) x GBM + w x MLP 로 바꾼다.
예측 제어 설정은 Park/e60 이 학습 구간에서 고른 값이다. 대표 값은 Park/e38 (w = 0.5, 1.0), Park/e35 (w = 0) 을 쓴다.

    python run_e61.py run --part <i> --nparts <n> --data .. --surrogate .. --mlp <darrow_surrogate_mlp.npz> --e60 <결과>/e60 --out <결과>/e61
    python run_e61.py report --out <결과>/e61 --e60 <결과>/e60 --e38 <결과>/e38 --e35 <결과>/e35 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from mlp_surrogate import MixedSurrogate, MLPSurrogate  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key  # noqa: E402

WS = {'w05': 0.5, 'w10': 1.0}   # w = 0.5 가 주 판정 (Park/e38 과 같음)


def gather(d, pat):
    res = {}
    for p in sorted(Path(d).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def run(a):
    sel = json.loads((Path(a.e60) / 'pick.json').read_text())
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    mlp = MLPSurrogate(a.mlp)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for w in list(WS)[a.part::a.nparts]:
        model = MixedSurrogate(dyn, mlp, WS[w])
        act = ShortMPC(env, model, margin=sel['margin'], lam=sel['lam'], objective='excess')
        out[f'mpc_{w}'] = evaluate(env, splits['val'], act, shield=MinAirBrake(model, 3.5, 1, 'others', **BRAKE))
        print(w, out[f'mpc_{w}'], flush=True)
        (Path(a.out) / f'e61_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    sel = json.loads((Path(a.e60) / 'pick.json').read_text())
    m = gather(a.out, 'e61_part*.json')
    m['mpc_w00'] = json.loads((Path(a.e60) / 'val.json').read_text())['mpc_brake']
    rep = gather(a.e38, 'e38_part*.json')
    rep.update({k.replace('orig_', 'fcsrl_w00_'): v for k, v in gather(a.e35, 'e35_part*.json').items() if k.startswith('orig_')})
    g = lambda w, k: np.array([rep[f'fcsrl_{w}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    L = [f"검증 40일 x 구성원 5개 = 200 에피소드. 환경은 GBM 그대로, 차단과 예측 제어의 예측만 (1 - w) x GBM + w x MLP. 차단 위 예측 제어는 Park/e60 이 고른 NH4 여유 {sel['margin']}, 공기 가중치 {sel['lam']}.", '',
         '| 방법 | w | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % |', '| --- | --- | --- | --- | --- | --- |']
    for w, wn in (('w00', '0'), ('w05', '0.5'), ('w10', '1.0')):
        L.append(f"| 대표 (묶음 5개 평균) | {wn} | {g(w, 'air_m3').mean():,.0f} | {g(w, 'cost').mean():.3f} | {g(w, 'nh4_viol_pct').mean():.2f} | {g(w, 'do_viol_pct').mean():.2f} |")
        r = m[f'mpc_{w}']
        L.append(f"| 차단 위 예측 제어 | {wn} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} |")
    dr = g('w05', 'cost') - g('w00', 'cost')
    dm = m['mpc_w05']['cost'] - m['mpc_w00']['cost']
    lv = g('w05', 'cost')
    h1, h2 = dr < dm, lv < m['mpc_w05']['cost']
    L += ['', '## 판정 (w = 0.5)', '', '| 묶음 | 대표 w0 | 대표 w0.5 | 대표 증가 | 대표 증가가 작음 | 대표 w0.5 가 낮음 |', '| --- | --- | --- | --- | --- | --- |']
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {g('w00', 'cost')[i]:.3f} | {lv[i]:.3f} | {dr[i]:+.3f} | {'예' if h1[i] else '아니오'} | {'예' if h2[i] else '아니오'} |")
    L += ['', f"- H1 (모델이 틀릴 때 대표가 덜 나빠지는가): 차단 위 예측 제어 증가 {dm:+.3f}, 대표 증가가 더 작은 묶음 {h1.sum()}/5, 평균 {dr.mean():+.3f}: {'지지' if h1.sum() >= 4 and dr.mean() < dm else '지지 안 됨'}",
          f"- H2 (모델이 틀릴 때 대표의 비용이 더 낮은가): 낮은 묶음 {h2.sum()}/5, 평균 {lv.mean():.3f} 대 {m['mpc_w05']['cost']:.3f}: {'지지' if h2.sum() >= 4 and lv.mean() < m['mpc_w05']['cost'] else '지지 안 됨'}",
          f"- 참고 증가 차이 (대표 빼기 예측 제어) {ci(dr - dm, '{:+.3f}')}",
          f"- 참고 w = 1.0: 대표 {g('w10', 'cost').mean():.3f}, 차단 위 예측 제어 {m['mpc_w10']['cost']:.3f}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=2)
    for k in ('data', 'surrogate', 'mlp', 'e60', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e60', 'e38', 'e35', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
