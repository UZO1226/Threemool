"""동일 목표 시각에서 예측 길이와 초기 NH4/DO 정보의 효과를 분리한다."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e27'))
from diagnose import arrays, load_model, routing_frame, metrics, sha, write, MODES, ROOT
from trial import pack, replay

HORIZONS = (1, 12, 36, 72, 144)


@torch.no_grad()
def endpoints(model, x, aux, targets, horizon, initial='both', free=None):
    """각 목표의 시작 상태만 주입한다. 시작 이후 수질 정답은 읽지 않는다."""
    state, exog = pack(x)
    state = torch.tensor(state).float()
    exog = torch.tensor(exog).float()
    extra = torch.tensor(aux).float()
    origins = np.asarray(targets)-horizon
    assert origins.min() >= 12 and np.max(targets) <= 287
    current = state[:, origins].clone()
    if initial != 'both':
        assert origins.min() >= 13 and free is not None
        current = torch.tensor(free[:, origins-13]).float()
        if initial == 'nh4':
            current[..., 0] = state[:, origins, 0]
        elif initial == 'do':
            current[..., 1] = state[:, origins, 1]
        else:
            assert initial == 'none'
    n, t, _ = current.shape
    current = current.reshape(-1, 2)
    for offset in range(horizon):
        indices = origins+offset
        current = model(current, exog[:, indices].reshape(-1, 6), extra[:, indices].reshape(-1, 4, 2))
    assert torch.isfinite(current).all()
    return current.reshape(n, t, 2).numpy()


def analyze(pred, truth):
    return metrics(pred, truth)


def main(out):
    if out.exists():
        raise ValueError('Refusing overwrite')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data_path = HERE.parents[1]/'data/avedore/aved_5min.pkl'
    raw_path = data_path.with_name('aved_raw.csv')
    e27_path = ROOT/'e27/diagnosis.json'
    previous = json.loads(e27_path.read_text())
    old_one = json.loads((HERE.parent/'e26/evaluation.json').read_text())
    data = pd.read_pickle(data_path)
    days = previous['days']
    assert not set(days['inner']+days['val']) & set(data['days']['test'])
    xs = {s: arrays(dict(data=data['data'], days=days), s) for s in days}
    frame = routing_frame(raw_path, end='2024-02-29')
    aux = {s: np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in days[s]]) for s in days}
    sources = [Path(__file__), HERE.parent/'e27/diagnose.py', HERE.parent/'e12/trial.py',
               HERE.parent/'e11/routing_trial.py', HERE.parent/'e10/history_trial.py']
    hashes = {str(p): sha(p) for p in (data_path, raw_path, e27_path, *sources)}
    for p in (data_path, raw_path):
        assert hashes[str(p)] == previous['input_sha256'][str(p)]
    report = dict(started_at=datetime.now(timezone.utc).isoformat(),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  config=dict(horizons=HORIZONS, intervention_horizon=36, seeds=[0, 1], modes=MODES,
                              early_targets=[49, 156], late_targets=[157, 287]),
                  days=days, hashes=hashes, rows=[], clock_rows=[], interventions=[], checks=[],
                  test_used=False, training_performed=False, model_replaced=False,
                  interpretation='observational forecast diagnosis with measured origin states, not autonomous daily control')
    for mode in MODES:
        for seed in (0, 1):
            path = (ROOT/'e12/trial'/f'multi_mse_s{seed}.pt' if mode == 'start_multi'
                    else ROOT/'e14/trial'/f'{mode}_s{seed}.pt')
            assert sha(path) == previous['model_sha256'][str(path)]
            hashes[str(path)] = sha(path)
            model = load_model(path)
            cache_path = ROOT/'e27'/f'{mode}_s{seed}.npz'
            hashes[str(cache_path)] = sha(cache_path)
            cached = np.load(cache_path)
            for split in days:
                x, extra = xs[split], aux[split]
                truth_all = np.stack((x['nh4'], x['do']), -1)
                late = np.arange(157, 288)
                truth = truth_all[:, late]
                free = cached[split]
                np.testing.assert_array_equal(free, replay(model, x, extra, 'with_routing', 275))
                report['rows'].append(dict(mode=mode, seed=seed, split=split, horizon='free',
                    target_times='13:05~23:55', **analyze(free[:, late-13], truth)))
                saved = dict(targets=late, truth=truth, free=free[:, late-13])
                for horizon in HORIZONS:
                    targets = np.arange(13 if horizon == 1 else 49 if horizon <= 36 else 157, 288)
                    pred = endpoints(model, x, extra, targets, horizon)
                    pl = pred[:, targets >= 157]
                    report['rows'].append(dict(mode=mode, seed=seed, split=split, horizon=horizon,
                        target_times='13:05~23:55', **analyze(pl, truth)))
                    saved[f'h{horizon}'] = pl
                    if horizon <= 36:
                        for label, mask in (('early', (targets >= 49) & (targets <= 156)), ('late', targets >= 157)):
                            report['clock_rows'].append(dict(mode=mode, seed=seed, split=split, horizon=horizon,
                                clock=label, **analyze(pred[:, mask], truth_all[:, targets[mask]])))
                    if horizon == 1 and split == 'val':
                        old = next(v for v in old_one['evaluation'] if v['mode'] == mode and v['seed'] == seed and v['reset_steps'] == 1)
                        now = analyze(pred, truth_all[:, targets])
                        for key in ('tp', 'fp', 'fn', 'tn'):
                            assert now[key] == old['nh4'][key]
                        np.testing.assert_allclose(pred, replay(model, x, extra, 'with_routing', 1), atol=1e-5, rtol=1e-5)
                    # 하나의 목표만 독립 실행해 그 시작 이후 정답 교란에 불변인지 확인한다.
                    target = np.array([200])
                    tiny = {k: v[:2].copy() for k, v in x.items()}
                    expected = endpoints(model, tiny, extra[:2], target, horizon)
                    tiny['nh4'][:, 201-horizon:] += 1000
                    tiny['do'][:, 201-horizon:] += 1000
                    np.testing.assert_array_equal(expected, endpoints(model, tiny, extra[:2], target, horizon))
                for initial in ('none', 'nh4', 'do', 'both'):
                    pred = endpoints(model, x, extra, late, 36, initial, free)
                    if initial == 'none':
                        np.testing.assert_allclose(pred, free[:, late-13], atol=1e-4, rtol=1e-4)
                        assert analyze(pred, truth)['fp'] == analyze(free[:, late-13], truth)['fp']
                    if initial == 'both':
                        np.testing.assert_allclose(pred, saved['h36'], atol=1e-5, rtol=1e-5)
                    report['interventions'].append(dict(mode=mode, seed=seed, split=split, initial=initial,
                        horizon=36, **analyze(pred, truth)))
                    saved[f'initial_{initial}'] = pred
                np.savez_compressed(out/f'{mode}_s{seed}_{split}.npz', **saved)
                report['checks'].append(dict(mode=mode, seed=seed, split=split,
                    cache_exact=True, origin_future_independent=True, free_continuation_matches=True,
                    both_matches_h36=True, one_step_e26_match=split == 'val'))
                print(mode, seed, split, 'done', flush=True)
    assert all(sha(p) == h for p, h in hashes.items())
    report['hashes_unchanged'] = True
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    write(out/'diagnosis.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
