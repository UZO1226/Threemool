"""Park/e43 공기량을 맞춘 비교 (Park 작성). 규칙은 README.md 를 따른다.
대표 (FCSRL 4 x 16 앙상블 + 대표 차단) 의 학습 구간 공기량에 비학습 기준 두 개 (ABAC, 차단만) 와 참고 기준 (기본 PPO-Lag 공기 배율) 을
학습 구간에서 맞춘 뒤, 검증 구간에서 같은 공기량의 비용을 비교한다.

    python run_e43.py rep-train --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --out <결과>/e43
    python run_e43.py abac-grid --part <i> --nparts <n> --data .. --surrogate .. --out <결과>/e43
    python run_e43.py brake-train --delta <d> --data .. --surrogate .. --out <결과>/e43
    python run_e43.py ppo-train --data .. --surrogate .. --runs <결과> --out <결과>/e43
    python run_e43.py pick --out <결과>/e43 --md select.md
    python run_e43.py val --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --out <결과>/e43
    python run_e43.py report --out <결과>/e43 --e35 <결과>/e35 --md results.md
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
from agent import METHODS, ObsNormalizer  # noqa: E402
from baselines_darrow import ABAC  # noqa: E402
from darrow_env import BatchEnv, Dynamics, air_to_action, evaluate, load_days  # noqa: E402
from min_brake import E15_MARGINS, EnsemblePolicy, MinAirBrake, ScaledPolicy  # noqa: E402
from run_e38 import BASE_PPO, BRAKE, COMBOS, ci, key, load  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

SP = tuple(round(1.0 + 0.1 * i, 1) for i in range(31))          # 1.0 ~ 4.0
KP = (0.0, 500.0, 1000.0, 2000.0)                                # Yoo/e68 과 같은 격자
KI = (400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0, 25600.0, 1e6)
GRID = list(itertools.product(SP, KP, KI))
DELTAS = (0.0, 0.1, 0.2, 0.3)
FACTORS = (0.80, 0.85, 0.90, 0.95, 1.00)
TOL = 0.02     # 학습 구간 맞춤 허용폭
VAL_TOL = 0.03  # 검증 구간 판정에 필요한 공기량 차이


def setup(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    return splits, BatchEnv(day_data, dyn), dyn


def brake_only(dyn, delta):
    kw = dict(BRAKE, margins=[(u, m + 0.1 + delta) for u, m in E15_MARGINS])
    return MinAirBrake(dyn, 3.5, 1, 'others', **kw)


def zero(obs):
    return np.full(len(obs), air_to_action(0.0))


def ppo_default(runs, s):
    ag = METHODS['ppo'](11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(Path(runs) / 'dw_runs' / f'ppo_s{s}' / 'model_final')
    return lambda obs: agent_act(ag, obs)


def rep(runs, c):
    return EnsemblePolicy([lambda obs, g=load(runs, 'fcsrl', s): agent_act(g, obs) for s in c])


def save(a, name, out):
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / name).write_text(json.dumps(out, indent=2))


def rep_train(a):
    splits, env, dyn = setup(a)
    out = {}
    for c in COMBOS[a.part::a.nparts]:
        out[key(c)] = evaluate(env, splits['train'][-40:], rep(a.runs, c), shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        print(c, out[key(c)], flush=True)
        save(a, f'rep_train_part{a.part}.json', out)


def abac_grid(a):
    splits, env, _ = setup(a)
    out = {}
    for sp, kp, ki in GRID[a.part::a.nparts]:
        out[f'{sp}_{kp}_{ki}'] = evaluate(env, splits['train'][-40:], ABAC(env, sp, kp, ki))
    save(a, f'abac_train_part{a.part}.json', out)


def brake_train(a):
    splits, env, dyn = setup(a)
    r = evaluate(env, splits['train'][-40:], zero, shield=brake_only(dyn, a.delta))
    print(a.delta, r, flush=True)
    save(a, f'brake_train_d{a.delta:.1f}.json', r)


def ppo_train(a):
    splits, env, _ = setup(a)
    out = {}
    for f in FACTORS:
        for s in range(5):
            out[f'{f:.2f}_{s}'] = evaluate(env, splits['train'][-40:], ScaledPolicy(ppo_default(a.runs, s), f))
        print(f, np.mean([out[f'{f:.2f}_{s}']['air_m3'] for s in range(5)]), flush=True)
    save(a, 'ppo_train.json', out)


def gather(out, pat):
    res = {}
    for p in sorted(Path(out).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def pick(a):
    o = Path(a.out)
    rt = gather(o, 'rep_train_part*.json')
    target = np.mean([rt[key(c)]['air_m3'] for c in COMBOS])
    ab = gather(o, 'abac_train_part*.json')
    assert len(ab) == len(GRID) and len(rt) == len(COMBOS)
    near = {k: v for k, v in ab.items() if abs(v['air_m3'] / target - 1) <= TOL}
    if near:
        ka = min(near, key=lambda k: (near[k]['cost'], near[k]['air_m3']))
    else:
        ka = min(ab, key=lambda k: abs(ab[k]['air_m3'] - target))
    sp, kp, ki = map(float, ka.split('_'))
    edge = [n for n, v, g in (('sp', sp, SP), ('kp', kp, KP), ('ki', ki, KI)) if (v == max(g) and n != 'ki') or (v == min(g) and n != 'kp')]
    br = {d: json.loads((o / f'brake_train_d{d:.1f}.json').read_text()) for d in DELTAS}
    kd = min(DELTAS, key=lambda d: (abs(br[d]['air_m3'] - target), d))
    pp = json.loads((o / 'ppo_train.json').read_text())
    pair = {f: np.mean([pp[f'{f:.2f}_{s}']['air_m3'] for s in range(5)]) for f in FACTORS}
    pc = {f: np.mean([pp[f'{f:.2f}_{s}']['cost'] for s in range(5)]) for f in FACTORS}
    kf = min(FACTORS, key=lambda f: (abs(pair[f] - target), f))
    sel = {'target_train_air': target, 'abac': [sp, kp, ki], 'abac_within_tol': bool(near), 'abac_edge': edge, 'delta': kd, 'factor': kf}
    (o / 'pick.json').write_text(json.dumps(sel, indent=2))
    L = [f'학습 구간 마지막 40일 x 구성원 5개 = 200 에피소드. 맞출 공기량 = 대표 (묶음 5개 평균) 의 학습 구간 공기량 {target:,.0f} Nm3/일.', '',
         '| 묶음 | 대표 학습 구간 공기량 | 하루 비용 |', '| --- | --- | --- |']
    L += [f"| {key(c)} | {rt[key(c)]['air_m3']:,.0f} | {rt[key(c)]['cost']:.3f} |" for c in COMBOS]
    L += ['', f'## ABAC 격자 {len(GRID)} 개 (설정값 1.0 ~ 4.0, Kp {KP}, Ki {KI})', '',
          f'공기량이 맞출 값의 ±{TOL * 100:.0f}% 안인 조합 {len(near)} 개 가운데 비용이 가장 낮은 것을 고른다.', '',
          '| 설정값 | Kp | Ki | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % |', '| --- | --- | --- | --- | --- | --- | --- |']
    for k in sorted(near or {ka: ab[ka]}, key=lambda k: ab[k]['cost'])[:10]:
        r = ab[k]
        s_, p_, i_ = k.split('_')
        L.append(f"| {s_}{' (고름)' if k == ka else ''} | {p_} | {i_} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} |")
    L += ['', f"- 고른 ABAC: 설정값 {sp}, Kp {kp}, Ki {ki}. 허용폭 안: {'예' if near else '아니오 (가장 가까운 공기량)'}. 격자 끝: {', '.join(edge) or '없음'}",
          f"- 참고 ABAC 격자 전체의 가장 낮은 비용 {min(v['cost'] for v in ab.values()):.3f} (공기량 {ab[min(ab, key=lambda k: ab[k]['cost'])]['air_m3']:,.0f})", '',
          '## 차단만 (정책 없음, 대표 차단의 NH4 여유 + delta)', '', '| delta | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % |', '| --- | --- | --- | --- | --- |']
    L += [f"| {d:.1f}{' (고름)' if d == kd else ''} | {br[d]['air_m3']:,.0f} | {br[d]['cost']:.3f} | {br[d]['nh4_viol_pct']:.2f} | {br[d]['do_viol_pct']:.2f} |" for d in DELTAS]
    L += ['', '## 참고: 기본 PPO-Lag (차단 없음) 공기 배율, seed 5개 평균', '', '| 배율 | 공기량 | 하루 비용 |', '| --- | --- | --- |']
    L += [f"| {f:.2f}{' (고름)' if f == kf else ''} | {pair[f]:,.0f} | {pc[f]:.3f} |" for f in FACTORS]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def val_jobs(sel):
    sp, kp, ki = sel['abac']
    jobs = [('abac', sp, kp, ki)] + [('abac_curve', s, kp, ki) for s in SP if s != sp]
    jobs += [('brake', sel['delta'])] + [('brake_curve', d) for d in DELTAS if d != sel['delta']]
    jobs += [('ppo', sel['factor'], s) for s in range(5)]
    return jobs


def val(a):
    sel = json.loads((Path(a.out) / 'pick.json').read_text())
    splits, env, dyn = setup(a)
    jobs = val_jobs(sel)
    # 오래 걸리는 차단 평가를 part 마다 고르게 나눈다
    slow = [j for j in jobs if j[0].startswith('brake')]
    fast = [j for j in jobs if not j[0].startswith('brake')]
    mine = slow[a.part::a.nparts] + fast[a.part::a.nparts]
    out = {}
    for j in mine:
        if j[0].startswith('abac'):
            r = evaluate(env, splits['val'], ABAC(env, *j[1:]))
        elif j[0].startswith('brake'):
            r = evaluate(env, splits['val'], zero, shield=brake_only(dyn, j[1]))
        else:
            r = evaluate(env, splits['val'], ScaledPolicy(ppo_default(a.runs, j[2]), j[1]))
        out['|'.join(map(str, j))] = r
        print(j, r, flush=True)
        save(a, f'val_part{a.part}.json', out)


def interp(curve, x):
    """(공기량, 비용) 점들에서 공기량 x 의 비용 (직선 보간, 범위 밖이면 nan)."""
    pts = sorted(curve)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1e-9)
    return float('nan')


def report(a):
    sel = json.loads((Path(a.out) / 'pick.json').read_text())
    v = gather(a.out, 'val_part*.json')
    e35 = gather(a.e35, 'e35_part*.json')
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')
    R = {k: np.array([e35[f'orig_{key(c)}'][k] for c in COMBOS]) for k in keys}
    sp, kp, ki = sel['abac']
    A = v[f'abac|{sp}|{kp}|{ki}']
    B = v[f"brake|{sel['delta']}"]
    P = {k: np.mean([v[f"ppo|{sel['factor']}|{s}"][k] for s in range(5)]) for k in keys}
    ra = R['air_m3'].mean()
    row = lambda n, m: f"| {n} | {m['air_m3']:,.0f} | {(m['air_m3'] / ra - 1) * 100:+.1f}% | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} |"  # noqa: E731
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 기준의 설정은 학습 구간에서 대표의 공기량에 맞춰 골랐다 ([select.md](select.md)).', '',
         '| 방법 | 공기량 Nm3/일 | 대표 대비 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % |', '| --- | --- | --- | --- | --- | --- | --- |',
         row('대표 (FCSRL 4 x 16 + 대표 차단, 묶음 5개 평균, e35 값)', {k: R[k].mean() for k in keys}),
         row(f'ABAC (설정값 {sp}, Kp {kp:g}, Ki {ki:g}), 대리 모델 없음', A),
         row(f"차단만 (정책 없음, 여유 + {sel['delta']:.1f})", B),
         row(f"참고: 기본 PPO-Lag 공기 x {sel['factor']:.2f}, 차단 없음, seed 5개 평균", P),
         f"| 참고: 기본 PPO-Lag, 차단 없음 (공식 표) | {BASE_PPO['air_m3']:,.0f} | {(BASE_PPO['air_m3'] / ra - 1) * 100:+.1f}% | {BASE_PPO['cost']:.2f} | | | |"]
    L += ['', '## 판정', '', '| 묶음 | 대표 공기량 | 대표 비용 | ABAC 보다 비용 낮음 | 차단만보다 비용 낮음 |', '| --- | --- | --- | --- | --- |']
    okA, okB = R['cost'] < A['cost'], R['cost'] < B['cost']
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {R['air_m3'][i]:,.0f} | {R['cost'][i]:.3f} | {'예' if okA[i] else '아니오'} | {'예' if okB[i] else '아니오'} |")
    for h, name, m, ok in (('H1', 'ABAC', A, okA), ('H2', '차단만', B, okB)):
        gap = m['air_m3'] / ra - 1
        if abs(gap) > VAL_TOL:
            verdict = f'판정 불가 (검증 공기량 차이 {gap * 100:+.1f}% 가 ±{VAL_TOL * 100:.0f}% 밖)'
        else:
            verdict = '지지' if ok.sum() >= 4 and R['cost'].mean() < m['cost'] else '지지 안 됨'
        L.append(f"- {h} 판정 ({name}): 공기량 차이 {gap * 100:+.1f}%, 비용이 낮은 묶음 {ok.sum()}/5, 평균 {R['cost'].mean():.3f} 대 {m['cost']:.3f}: {verdict}")
        L.append(f"- 참고 {h} 비용 차이 (대표 빼기 {name}) {ci(R['cost'] - m['cost'], '{:+.3f}')}")
    # 참고: 검증 구간 곡선에서 대표 공기량의 비용
    ac = [(A['air_m3'], A['cost'])] + [(r['air_m3'], r['cost']) for k, r in v.items() if k.startswith('abac_curve')]
    bc = [(B['air_m3'], B['cost'])] + [(r['air_m3'], r['cost']) for k, r in v.items() if k.startswith('brake_curve')]
    L += ['', '## 참고: 검증 구간 곡선 (판정 아님)', '',
          f'같은 Kp, Ki 에서 설정값만 바꾼 ABAC {len(ac)} 점, 차단만 delta {len(bc)} 점을 검증 구간에서 돌려, 대표 공기량 {ra:,.0f} 에서의 비용을 직선 보간했다.', '',
          f'- ABAC: {interp(ac, ra):.3f}', f'- 차단만: {interp(bc, ra):.3f}', '',
          '| 기준 | 공기량 | 하루 비용 |', '| --- | --- | --- |']
    L += [f'| ABAC | {x:,.0f} | {y:.3f} |' for x, y in sorted(ac)] + [f'| 차단만 | {x:,.0f} | {y:.3f} |' for x, y in sorted(bc)]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    cmds = {'rep-train': rep_train, 'abac-grid': abac_grid, 'brake-train': brake_train, 'ppo-train': ppo_train, 'val': val}
    for n in cmds:
        q = sub.add_parser(n)
        for k in ('data', 'surrogate', 'out'):
            q.add_argument(f'--{k}', required=True)
        q.add_argument('--runs')
        q.add_argument('--part', type=int, default=0)
        q.add_argument('--nparts', type=int, default=1)
        q.add_argument('--delta', type=float, default=0.0)
    k = sub.add_parser('pick')
    for x in ('out', 'md'):
        k.add_argument(f'--{x}', required=True)
    r = sub.add_parser('report')
    for x in ('out', 'e35', 'md'):
        r.add_argument(f'--{x}', required=True)
    a = p.parse_args()
    {**cmds, 'pick': pick, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
