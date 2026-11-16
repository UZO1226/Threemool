"""e54 준비: 날짜 묶음을 빼고 학습한 NH4 불확실성 앙상블을 만들고, 정책 학습 전에 불확실성의 질을 점검하고 kappa 를 정한다.

학습 188일을 날짜순으로 5묶음(37 ~ 38일)으로 나누고, 구성원 m 은 묶음 m 을 뺀 나머지 날짜의 모든 행으로 NH4 변화량 GBM 을
학습한다. 하이퍼파라미터와 단조 제약은 e35 수정 모델의 NH4 구성원 0 과 같다. 이 앙상블은 학습 비용의 여유를 계산할 때만
쓰고, 상태 전이는 e35 수정 모델 그대로다. 검증 날짜는 kappa 선택에 쓰지 않는다 (참고값만 기록).

    python build.py --prepared /home/jio/fcsrl_src/results/e30/prepared \
        --corrected /home/jio/fcsrl_src/results/e35/zone_fix/corrected_surrogate.pkl --out /home/jio/fcsrl_src/results/e54/unc
"""
import argparse
from datetime import datetime, timezone
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
sys.path.insert(0, str(HERE.parent / 'e44'))
from common import save, sha  # noqa: E402
from surrogate_dw import build_arrays  # noqa: E402
from darrow_env import Dynamics, load_days, all_episodes, action_to_air, NH4_MAX  # noqa: E402
from command import CommandEnv  # noqa: E402

TARGET = 5.62  # 학습 날짜 기록 운전의 실제 NH4 위반 (e33, e52, e53 과 같은 기준)
KAPPAS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100)


def blocks(train_days, n=5):
    return [list(b) for b in np.array_split(np.array(train_days), n)]


def member_sd(models, x):
    return np.stack([m.predict(x) for m in models]).std(0)


def one_step(models, x, y):
    """한 스텝 평가: 구성원 표준편차와 구성원 평균의 절대 오차, 그 순위 상관"""
    pred = np.stack([m.predict(x) for m in models])
    sd, err = pred.std(0), np.abs(pred.mean(0) - y)
    rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    top = sd >= np.quantile(sd, .9)
    return {'sd_mean': float(sd.mean()), 'sd_p90': float(np.quantile(sd, .9)), 'abs_err_mean': float(err.mean()),
            'spearman_sd_abs_err': float(np.corrcoef(rank(sd), rank(err))[0, 1]),
            'abs_err_top10pct_sd': float(err[top].mean()), 'abs_err_rest': float(err[~top].mean())}


def replay(env, dyn, unc, episodes, deltas=(0.45,), kappas=KAPPAS):
    """기록 운전 재생: 여유 포함 NH4 비용 (kappa 별, 고정 여유 별)"""
    env.reset(episodes)
    n = len(env.episodes)
    by_k, by_d, sds = {k: np.zeros(n) for k in kappas}, {d: np.zeros(n) for d in deltas}, []
    for _ in range(env.T):
        t, a = env.t, env.historical_action()
        zone = action_to_air(a)[:, None] * env.x['frac'][:, t]
        x = np.column_stack([env.state, env.x['exo'][:, t], zone / 1000.0, env.prev_zone / 1000.0])
        sd = member_sd(unc, x)
        sds.append(sd)
        env.step(a)
        nh = env.state[:, 0]
        for k in kappas:
            by_k[k] += (nh + k * sd) > NH4_MAX
        for d in deltas:
            by_d[d] += (nh + d) > NH4_MAX
    s = np.concatenate(sds)
    return {'sd_mean': float(s.mean()), 'sd_p90': float(np.quantile(s, .9)),
            'kappa_cost': {str(k): float(v.mean()) for k, v in by_k.items()},
            'delta_cost': {str(d): float(v.mean()) for d, v in by_d.items()}}


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    data = pd.read_pickle(a.prepared / 'data.pkl')
    train_days, val_days = list(data['days']['train']), list(data['days']['val'])
    with a.corrected.open('rb') as f:
        corrected = pickle.load(f)
    base = corrected['models'][0][0]
    params = base.get_params()
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(),
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
                'input_sha256': {'data.pkl': sha(a.prepared / 'data.pkl'), 'corrected_surrogate.pkl': sha(a.corrected)},
                'source_sha256': {p.name: sha(p) for p in HERE.glob('*.py')}, 'params': {k: str(v) for k, v in params.items()},
                'split': 'train days in 5 chronological blocks, member m leaves block m out, all rows of remaining days',
                'test_evaluated': False, 'kappa_selection': 'train days only'}
    bl = blocks(train_days)
    unc, oof_x, oof_y, oof_m = [], [], [], []
    for m, held in enumerate(bl):
        keep = [d for d in train_days if d not in set(held)]
        x, y = build_arrays(data['data'], keep)
        g = HistGradientBoostingRegressor(**params).fit(x, y[:, 0])
        unc.append(g)
        hx, hy = build_arrays(data['data'], held)
        oof_x.append(hx), oof_y.append(hy[:, 0]), oof_m.append(np.full(len(hx), m))
        print('FIT', m, held[0], held[-1], len(keep), 'days', g.n_iter_, 'iters', flush=True)
    manifest['blocks'] = [[b[0], b[-1], len(b)] for b in bl]
    # 한 스텝 점검: 학습 날짜 전체 (각 날짜는 구성원 하나가 못 봄), 학습 날짜 빠진 묶음만 (해당 구성원 예측 오차), 검증 날짜
    xt, yt = build_arrays(data['data'], train_days)
    xv, yv = build_arrays(data['data'], val_days)
    orig = [corrected['models'][m][0] for m in range(5)]
    oof_err = np.concatenate([np.abs(unc[m].predict(x) - y) for m, x, y in zip(range(5), oof_x, oof_y)])
    in_err = np.concatenate([np.abs(np.mean([unc[k].predict(x) for k in range(5) if k != m], 0) - y)
                             for m, x, y in zip(range(5), oof_x, oof_y)])
    check = {'one_step': {'original': {'train': one_step(orig, xt, yt[:, 0]), 'val': one_step(orig, xv, yv[:, 0])},
                          'day_split': {'train': one_step(unc, xt, yt[:, 0]), 'val': one_step(unc, xv, yv[:, 0])}},
             'held_out_block_abs_err': float(oof_err.mean()), 'seen_members_abs_err_same_rows': float(in_err.mean())}
    splits, days = load_days(a.prepared / 'data.pkl')
    dyn = Dynamics(a.corrected)
    env = CommandEnv(days, dyn, 'padded')
    check['replay'] = {'day_split': {sp: replay(env, dyn, unc, all_episodes(splits[sp], 5)) for sp in ('train', 'val')},
                       'original': {sp: replay(env, dyn, orig, all_episodes(splits[sp], 5)) for sp in ('train', 'val')}}
    tr = check['replay']['day_split']['train']['kappa_cost']
    ks = np.array(KAPPAS, float)
    cs = np.array([tr[str(k)] for k in KAPPAS])
    hi = int(np.argmax(cs >= TARGET))
    assert cs[hi] >= TARGET > cs[hi - 1]
    # 선형 보간 뒤 0.5 단위로 반올림해 고정, 그 값으로 다시 재생해 확인
    kappa = float(np.round((ks[hi - 1] + (TARGET - cs[hi - 1]) / (cs[hi] - cs[hi - 1]) * (ks[hi] - ks[hi - 1])) * 2) / 2)
    chosen = {sp: replay(env, dyn, unc, all_episodes(splits[sp], 5), kappas=(kappa,))['kappa_cost'][str(kappa)] for sp in ('train', 'val')}
    check['chosen'] = {'kappa': kappa, 'train_replay_cost': chosen['train'], 'val_replay_cost_reference': chosen['val'],
                       'fixed_delta_0.45_val_replay_cost_reference': check['replay']['day_split']['val']['delta_cost']['0.45']}
    with (a.out / 'nh4_uncertainty.pkl').open('wb') as f:
        pickle.dump({'models': unc, 'kappa': kappa, 'blocks': manifest['blocks']}, f)
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), model_sha256=sha(a.out / 'nh4_uncertainty.pkl'))
    save(a.out / 'manifest.json', manifest)
    save(a.out / 'check.json', check)
    save(HERE / 'check.json', check)
    print('CHOSEN', check['chosen'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--corrected', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
