"""Park/e41 틀린 차단 모델에 온라인 NH4 오차 보정 (Park 작성). 규칙은 README.md 를 따른다.
차단 예측은 Park/e38 의 섞음 w = 0.5 이고, Park/e31 의 온라인 보정 (bias = lam 0.8, 여유 배율 1.0) 을 켠다.
참고로 맞는 차단 (w = 0) 에도 같은 보정을 켠다. 정책은 FCSRL 대표, 구성원이 겹치지 않는 묶음 5개.

    python run_e41.py run --part <i> --nparts <n> --data <darrow_15min.pkl> --surrogate <gbm.pkl> --mlp <mlp.npz> --runs <결과> --out <결과>/e41
    python run_e41.py report --out <결과>/e41 --e35 <결과>/e35 --e38 <결과>/e38 --e40 <결과>/e40 --md results.md
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
from min_brake import E15_MARGINS, EnsemblePolicy, MinAirBrake  # noqa: E402
from mlp_surrogate import MixedSurrogate, MLPSurrogate  # noqa: E402
from run_e38 import BASE_PPO, COMBOS, ci, key, load  # noqa: E402
from train_darrow import agent_act  # noqa: E402

LAM = 0.8   # Park/e31 이 고른 값, 이 실험에서는 고르지 않는다
P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
WS = {'b05': 0.5, 'b00': 0.0}
COND = [(w, c) for w in WS for c in COMBOS]   # 틀린 차단 (주 판정) 을 먼저 돈다


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    mlp = MLPSurrogate(a.mlp)
    env = BatchEnv(day_data, dyn)
    out = {}
    for w, c in COND[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, 'fcsrl', s): agent_act(g, obs) for s in c])
        model = dyn if WS[w] == 0.0 else MixedSurrogate(dyn, mlp, WS[w])
        br = MinAirBrake(model, 3.5, 1, 'others', trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1,
                         plan=(4, 1.8, 500.0), bias=LAM)
        r = evaluate(env, splits['val'], ens, shield=br)
        out[f'fcsrl_{w}_{key(c)}'] = r
        print(w, c, r, flush=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f'e41_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    res = {}
    for p in sorted(Path(a.out).glob('e41_part*.json')):
        res.update(json.loads(p.read_text()))
    for p in sorted(Path(a.e38).glob('e38_part*.json')):
        res.update({k.replace('fcsrl_w05_', 'fcsrl_r05_'): v for k, v in json.loads(p.read_text()).items() if k.startswith('fcsrl_w05_')})
    for p in sorted(Path(a.e35).glob('e35_part*.json')):
        res.update({k.replace('orig_', 'fcsrl_r00_'): v for k, v in json.loads(p.read_text()).items() if k.startswith('orig_')})
    for p in sorted(Path(a.e40).glob('e40_part*.json')):
        res.update({k.replace('fcsrl_cal_', 'fcsrl_c05_'): v for k, v in json.loads(p.read_text()).items()})
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    get = lambda h, k: np.array([res[f'fcsrl_{h}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    lab = {'r00': '맞는 차단 (w = 0, e35)', 'b00': '맞는 차단 + 온라인 보정 (이 실험, 참고)',
           'r05': '틀린 차단 w = 0.5 (e38)', 'c05': '틀린 차단 + 여유 0.6 (e40)', 'b05': '틀린 차단 + 온라인 보정 (이 실험)'}
    L = [f'검증 40일 x 구성원 5개 = 200 에피소드. FCSRL 대표, 구성원이 겹치지 않는 묶음 5개 평균. 온라인 보정 lam {LAM}.', '',
         '| 차단 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |', '| --- | --- | --- | --- | --- | --- | --- |']
    for h in ('r00', 'b00', 'r05', 'c05', 'b05'):
        m = {k: get(h, k).mean() for k in keys}
        L.append(f"| {lab[h]} | {m['air_m3']:.0f} | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} | {m['intervention_pct']:.1f} |")
    L.append(f"| 기본 PPO-Lag, 차단 없음 (공식 표) | {BASE_PPO['air_m3']:.0f} | {BASE_PPO['cost']:.2f} | | | | |")
    ba, bc = get('b05', 'air_m3'), get('b05', 'cost')
    ok1 = bc < get('r05', 'cost')
    ok2 = (ba < BASE_PPO['air_m3']) & (bc < BASE_PPO['cost'])
    ok3 = (ba < get('c05', 'air_m3')) & (bc < get('c05', 'cost'))
    L += ['', '## 묶음별 (틀린 차단 + 온라인 보정)', '',
          '| 묶음 | 공기량 | 비용 | NH4 위반 % | H1 e38 보다 비용 낮음 | H2 기준선보다 둘 다 낮음 | H3 e40 보다 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for i, c in enumerate(COMBOS):
        r = res[f'fcsrl_b05_{key(c)}']
        L.append(f"| {key(c)} | {r['air_m3']:.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {'예' if ok1[i] else '아니오'} | {'예' if ok2[i] else '아니오'} | {'예' if ok3[i] else '아니오'} |")
    yn = lambda s: '지지' if s else '지지 안 됨'  # noqa: E731
    L += ['', f"- H1 판정: e38 보다 비용 낮은 묶음 {ok1.sum()}/5, 평균 {bc.mean():.3f} 대 {get('r05', 'cost').mean():.3f}: {yn(ok1.sum() >= 4 and bc.mean() < get('r05', 'cost').mean())}",
          f"- H2 판정: 기준선보다 둘 다 낮은 묶음 {ok2.sum()}/5, 공기량 {ba.mean():.0f} 대 {BASE_PPO['air_m3']:.0f}, 비용 {bc.mean():.3f} 대 {BASE_PPO['cost']:.2f}: "
          f"{yn(ok2.sum() >= 4 and ba.mean() < BASE_PPO['air_m3'] and bc.mean() < BASE_PPO['cost'])}",
          f"- H3 판정: e40 보다 둘 다 낮은 묶음 {ok3.sum()}/5, 공기량 {ba.mean():.0f} 대 {get('c05', 'air_m3').mean():.0f}, 비용 {bc.mean():.3f} 대 {get('c05', 'cost').mean():.3f}: "
          f"{yn(ok3.sum() >= 4 and ba.mean() < get('c05', 'air_m3').mean() and bc.mean() < get('c05', 'cost').mean())}"]
    for x, y in (('b05', 'r05'), ('b05', 'c05'), ('b05', 'r00'), ('b00', 'r00')):
        L.append(f"- 참고 {lab[x]} 빼기 {lab[y]}: 공기량 {ci(get(x, 'air_m3') - get(y, 'air_m3'), '{:+,.0f}')}, 비용 {ci(get(x, 'cost') - get(y, 'cost'), '{:+.3f}')}, "
                 f"NH4 위반 {ci(get(x, 'nh4_viol_pct') - get(y, 'nh4_viol_pct'), '{:+.2f}')}")
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
    for k in ('out', 'e35', 'e38', 'e40', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
