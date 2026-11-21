"""e73 사전 검사와 차단 margin 선택 (정책 평가 전).

1. 차단 없는 재평가가 e72 metrics.json (fcsrl_s0, ppolag_s0 의 dev 표본 행동 k = 0, 평균 행동) 을 정확히 재현한다.
2. margin = -inf (차단이 올리지 않음) 의 결과가 차단 없는 평가와 정확히 같다.
3. 정답 모델 (환경과 같은 구성원) 차단에서 해가 있는 스텝의 실제 NH4 가 목표 이하다 (dev 전체, 공기 0 정책, margin 0).
4. 공기 0 정책 + 차단 (다른 구성원 예측) 을 protocol.json 의 margin 격자에서 dev 에 돌려 규칙대로 margin 을 고른다.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval73 as V  # noqa: E402

P = json.loads((HERE / 'protocol.json').read_text())


def calib(margin):
    torch.set_num_threads(1)
    splits, env = V.make_env()
    res, cnt, _, _ = V.evaluate(env, splits['dev'], V.air0, margin)
    return margin, res, cnt


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    splits, env = V.make_env()
    out = {}
    # 1, 2
    for method in ('fcsrl', 'ppolag'):
        ref = json.loads((V.E72_RUNS / f'{method}_s0' / 'metrics.json').read_text())
        agent = V.load_agent(method, 0)
        torch.manual_seed(V.SAMPLE_SEED(0, 0))
        res, *_ = V.evaluate(env, splits['dev'], agent.act_sample, None)
        torch.manual_seed(V.SAMPLE_SEED(0, 0))
        res_inf, cnt_inf, *_ = V.evaluate(env, splits['dev'], agent.act_sample, -np.inf)
        det, *_ = V.evaluate(env, splits['dev'], agent.act_det, None)
        out[f'{method}_s0'] = dict(stoch0_same=res == ref['dev_stochastic_runs'][0], det_same=det == ref['dev'],
                                   inf_margin_same=res_inf == res, inf_margin_interventions=cnt_inf['intervention_pct'])
        print(method, out[f'{method}_s0'], flush=True)
        assert out[f'{method}_s0']['stoch0_same'] and out[f'{method}_s0']['det_same'] and out[f'{method}_s0']['inf_margin_same']
    # 3
    res, cnt, (states, _), log = V.evaluate(env, splits['dev'], V.air0, 0.0, members='same', keep_log=True)
    feas = bad = 0
    for t, (tgt, has, pred, raised) in enumerate(log):
        feas += int(has.sum())
        bad += int(((states[:, t, 0] > tgt + 1e-9) & has).sum())
    out['oracle'] = dict(feasible_steps=feas, violated_feasible_steps=bad, dev=res, shield=cnt)
    print('oracle', feas, bad, flush=True)
    assert bad == 0
    # 4
    grid = list(P['shield']['margin_grid'])
    rows = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for m, res, cnt in ex.map(calib, grid):
            rows[m] = dict(dev=res, shield=cnt)
            print(f"margin {m}: over {res['over_budget_steps']:.3f} air {res['air_m3']:.0f} int {cnt['intervention_pct']:.1f}", flush=True)
        ok = [m for m in grid if rows[m]['dev']['over_budget_steps'] <= 0.3]
        if ok and min(ok, key=lambda m: rows[m]['dev']['air_m3']) == min(grid):
            ext = [-2.5, -3.0, -3.5, -4.0]
            for m, res, cnt in ex.map(calib, ext):
                rows[m] = dict(dev=res, shield=cnt)
                print(f"margin {m}: over {res['over_budget_steps']:.3f} air {res['air_m3']:.0f} int {cnt['intervention_pct']:.1f}", flush=True)
            grid += ext
            ok = [m for m in grid if rows[m]['dev']['over_budget_steps'] <= 0.3]
        if ok and min(ok, key=lambda m: rows[m]['dev']['air_m3']) == max(grid):
            ext = [0.75, 1.0, 1.25, 1.5, 2.0]   # 2026-11-21 수정 (protocol.json amendment_20261121)
            for m, res, cnt in ex.map(calib, ext):
                rows[m] = dict(dev=res, shield=cnt)
                print(f"margin {m}: over {res['over_budget_steps']:.3f} air {res['air_m3']:.0f} int {cnt['intervention_pct']:.1f}", flush=True)
            grid += ext
            ok = [m for m in grid if rows[m]['dev']['over_budget_steps'] <= 0.3]
    chosen = min(ok, key=lambda m: rows[m]['dev']['air_m3']) if ok else None
    out['calibration'] = dict(grid=grid, rows={str(k): v for k, v in rows.items()}, chosen_margin=chosen)
    (a.out / 'checks.json').write_text(json.dumps(out, indent=1))
    print('CHOSEN', chosen, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=9)
    main(p.parse_args())
