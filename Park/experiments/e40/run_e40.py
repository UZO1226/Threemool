"""Park/e40 틀린 차단 모델의 여유 다시 맞추기 (Park 작성). 규칙은 README.md 를 따른다.
차단 예측은 Park/e38 의 섞음 w = 0.5 (공기 효과를 절반만 아는 모델) 이고, NH4 여유 P1 의 모든 칸에 delta 를 더한다.
고르기: 학습 구간 마지막 40일, FCSRL seed 0 ~ 3 앙상블, delta 0 / 0.2 / 0.4 / 0.6. 하루 비용이 가장 낮은 delta 를 고른다.
검증: 고른 delta 로 구성원이 겹치지 않는 묶음 5개. w = 0.5, delta = 0 의 검증 값은 Park/e38 을 쓴다.

    python run_e40.py select --delta <d> --data <darrow_15min.pkl> --surrogate <gbm.pkl> --mlp <mlp.npz> --runs <결과> --out <결과>/e40
    python run_e40.py pick --out <결과>/e40 --md select.md
    python run_e40.py run --part <i> --nparts <n> --data ... --surrogate ... --mlp ... --runs <결과> --out <결과>/e40
    python run_e40.py report --out <결과>/e40 --e35 <결과>/e35 --e38 <결과>/e38 --md results.md
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

DELTAS = (0.0, 0.2, 0.4, 0.6)
W = 0.5


def brake(dyn, mlp, delta):
    margins = [(u, m + 0.1 + delta) for u, m in E15_MARGINS]   # P1 + delta
    return MinAirBrake(MixedSurrogate(dyn, mlp, W), 3.5, 1, 'others', trim=True, margins=margins, do_limit=1.8,
                       recover=(3, 3.5), conflict=0.1, plan=(4, 1.8, 500.0))


def ensemble(runs, c):
    return EnsemblePolicy([lambda obs, g=load(runs, 'fcsrl', s): agent_act(g, obs) for s in c])


def setup(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    return splits, BatchEnv(day_data, dyn), dyn, MLPSurrogate(a.mlp)


def select(a):
    splits, env, dyn, mlp = setup(a)
    r = evaluate(env, splits['train'][-40:], ensemble(a.runs, (0, 1, 2, 3)), shield=brake(dyn, mlp, a.delta))
    print(a.delta, r, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'select_d{a.delta:.1f}.json').write_text(json.dumps(r, indent=2))


def pick(a):
    res = {d: json.loads((Path(a.out) / f'select_d{d:.1f}.json').read_text()) for d in DELTAS}
    best = min(DELTAS, key=lambda d: (res[d]['cost'], d))
    L = ['학습 구간 마지막 40일 x 구성원 5개 = 200 에피소드, FCSRL seed 0 ~ 3 앙상블, 차단 예측 섞음 w = 0.5.', '',
         '| delta | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 개입 % |', '| --- | --- | --- | --- | --- | --- |']
    for d in DELTAS:
        r = res[d]
        L.append(f"| {d:.1f}{' (고름)' if d == best else ''} | {r['air_m3']:.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {r['intervention_pct']:.1f} |")
    L += ['', f'- 고른 delta: {best:.1f} (하루 비용 최소, 같으면 작은 쪽)']
    (Path(a.out) / 'pick.json').write_text(json.dumps({'delta': best}))
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def run(a):
    delta = json.loads((Path(a.out) / 'pick.json').read_text())['delta']
    assert delta > 0, '고른 delta 가 0 이면 검증하지 않는다 (README)'
    splits, env, dyn, mlp = setup(a)
    out = {}
    for c in COMBOS[a.part::a.nparts]:
        r = evaluate(env, splits['val'], ensemble(a.runs, c), shield=brake(dyn, mlp, delta))
        out[f'fcsrl_cal_{key(c)}'] = r
        print(c, r, flush=True)
        (Path(a.out) / f'e40_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    delta = json.loads((Path(a.out) / 'pick.json').read_text())['delta']
    res = {}
    for p in sorted(Path(a.out).glob('e40_part*.json')):
        res.update(json.loads(p.read_text()))
    for p in sorted(Path(a.e38).glob('e38_part*.json')):
        res.update({k.replace('fcsrl_w05_', 'fcsrl_raw_'): v for k, v in json.loads(p.read_text()).items() if k.startswith('fcsrl_w05_')})
    for p in sorted(Path(a.e35).glob('e35_part*.json')):
        res.update({k.replace('orig_', 'fcsrl_w00_'): v for k, v in json.loads(p.read_text()).items() if k.startswith('orig_')})
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    get = lambda h, k: np.array([res[f'fcsrl_{h}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    lab = {'w00': '맞는 차단 (w = 0, e35 값)', 'raw': f'틀린 차단 w = 0.5, 여유 그대로 (e38 값)', 'cal': f'틀린 차단 w = 0.5, 여유 + {delta:.1f} (이 실험)'}
    L = [f'검증 40일 x 구성원 5개 = 200 에피소드. FCSRL 대표, 구성원이 겹치지 않는 묶음 5개 평균. 고른 delta = {delta:.1f}.', '',
         '| 차단 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |', '| --- | --- | --- | --- | --- | --- | --- |']
    for h in ('w00', 'raw', 'cal'):
        m = {k: get(h, k).mean() for k in keys}
        L.append(f"| {lab[h]} | {m['air_m3']:.0f} | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} | {m['intervention_pct']:.1f} |")
    L.append(f"| 기본 PPO-Lag, 차단 없음 (공식 표) | {BASE_PPO['air_m3']:.0f} | {BASE_PPO['cost']:.2f} | | | | |")
    L += ['', '## 묶음별', '', '| 묶음 | 여유 그대로 비용 | 다시 맞춤 공기량 | 다시 맞춤 비용 | H1 비용 낮음 | H2 기준선보다 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- |']
    raw_c, cal_c, cal_a = get('raw', 'cost'), get('cal', 'cost'), get('cal', 'air_m3')
    ok1 = cal_c < raw_c
    ok2 = (cal_a < BASE_PPO['air_m3']) & (cal_c < BASE_PPO['cost'])
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {raw_c[i]:.3f} | {cal_a[i]:.0f} | {cal_c[i]:.3f} | {'예' if ok1[i] else '아니오'} | {'예' if ok2[i] else '아니오'} |")
    s1 = ok1.sum() >= 4 and cal_c.mean() < raw_c.mean()
    s2 = ok2.sum() >= 4 and cal_a.mean() < BASE_PPO['air_m3'] and cal_c.mean() < BASE_PPO['cost']
    L += ['', f"- H1 판정: 비용이 여유 그대로보다 낮은 묶음 {ok1.sum()}/5, 평균 {cal_c.mean():.3f} 대 {raw_c.mean():.3f}: {'지지' if s1 else '지지 안 됨'}",
          f"- H2 판정: 기준선보다 둘 다 낮은 묶음 {ok2.sum()}/5, 공기량 {cal_a.mean():.0f} 대 {BASE_PPO['air_m3']:.0f}, 비용 {cal_c.mean():.3f} 대 {BASE_PPO['cost']:.2f}: {'지지' if s2 else '지지 안 됨'}"]
    for h in ('raw', 'w00'):
        L.append(f"- 참고 다시 맞춤 빼기 {lab[h]}: 공기량 {ci(cal_a - get(h, 'air_m3'), '{:+,.0f}')}, 비용 {ci(cal_c - get(h, 'cost'), '{:+.3f}')}, "
                 f"NH4 위반 {ci(get('cal', 'nh4_viol_pct') - get(h, 'nh4_viol_pct'), '{:+.2f}')}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select')
    s.add_argument('--delta', type=float, required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=4)
    for q in (s, o):
        for k in ('data', 'surrogate', 'mlp', 'runs', 'out'):
            q.add_argument(f'--{k}', required=True)
    k = sub.add_parser('pick')
    for x in ('out', 'md'):
        k.add_argument(f'--{x}', required=True)
    r = sub.add_parser('report')
    for x in ('out', 'e35', 'e38', 'md'):
        r.add_argument(f'--{x}', required=True)
    a = p.parse_args()
    {'select': select, 'pick': pick, 'run': run, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
