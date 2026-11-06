"""하루 길이 예측과 전체 및 조건별 거짓경보 제약을 대조한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e13'))
from experiment import load_model, routing_frame, arrays, pack, evaluate, gates, sha, write

MODES = ('short_mse', 'day_mse', 'day_global', 'day_group')
GROUPS = ('all', 'phase_0', 'phase_1', 'phase_2', 'temp_lower', 'temp_middle', 'temp_upper')
CFG = dict(horizon=275, short_reset=36, updates=300, batch=16, lr=1e-4,
           weight_decay=1e-5, grad_clip=1., seeds=[0, 1], sampling_seed_base=14000,
           calibration_seed=14999, rate_gradient_ratio=.5, thresholds=[4., 2.],
           fpr_cap=.1, dual_lr=1., dual_ema=.9, dual_initial=0.,
           dual_upper_bound=None, min_negative_count=64, checkpoint='last',
           primary_candidate='day_group', false_negative_proxy=False)


def group_masks(exog, bounds):
    phase, temp = exog[..., 1], exog[..., 2]
    return torch.stack([torch.ones_like(temp, dtype=torch.bool)] +
                       [phase == k for k in range(3)] +
                       [temp < bounds[0], (temp >= bounds[0]) & (temp < bounds[1]), temp >= bounds[1]], -1)


def group_terms(pred, target, exog, bounds, temperature):
    threshold = pred.new_tensor(CFG['thresholds'])
    negative = (target <= threshold).unsqueeze(-2) & group_masks(exog, bounds).unsqueeze(-1)
    count = negative.sum((0, 1))
    logits = ((pred - threshold) / temperature).unsqueeze(-2)
    proxy = (F.softplus(logits) * negative).sum((0, 1)) / count.clamp_min(1)
    fp = ((pred.detach() > threshold).unsqueeze(-2) & negative).sum((0, 1))
    return proxy, fp, count


def objectives(model, state, exog, extra, scale, bounds, reset):
    pred = state[:, 0]
    sequence = []
    for t in range(exog.shape[1]):
        if t % reset == 0:
            pred = state[:, t]
        pred = model(pred, exog[:, t], extra[:, t].reshape(-1, 4, 2))
        sequence.append(pred)
    sequence = torch.stack(sequence, 1)
    multi = ((sequence - state[:, 1:]) / scale).square().mean()
    teacher = model(state[:, :-1].reshape(-1, 2), exog.reshape(-1, 6), extra.reshape(-1, 4, 2))
    anchor = ((teacher - state[:, 1:].reshape(-1, 2)) / scale).square().mean()
    return multi, anchor, group_terms(sequence, state[:, 1:], exog, bounds, model.delta_sd)


def dual_step(multiplier, average, initialized, fp, count):
    valid = count >= CFG['min_negative_count']
    observed = fp.detach() / count.clamp_min(1)
    smoothed = torch.where(initialized, CFG['dual_ema'] * average + (1-CFG['dual_ema']) * observed, observed)
    average = torch.where(valid, smoothed, average)
    updated = (multiplier + CFG['dual_lr'] * (average - CFG['fpr_cap']) * valid).clamp_min(0)
    return updated.detach(), average.detach(), initialized | valid


def penalty(proxy, multiplier, count, mode):
    if mode not in ('day_global', 'day_group'):
        return proxy.sum() * 0
    if mode == 'day_global':
        proxy, multiplier, count = proxy[:1], multiplier[:1], count[:1]
    return (proxy * multiplier * (count >= CFG['min_negative_count'])).mean()


def day_batch(tensors, rng):
    days = rng.integers(0, len(tensors[0]), CFG['batch'])
    state, exog, extra = tensors
    return (state[days, 12:], exog[days, 12:-1], extra[days, 12:-1]), days.tobytes()


def tradeoff_gate(rows, candidate):
    checks, reductions = [], {}
    for base in ('start_multi', 'day_mse'):
        gains = []
        for seed in CFG['seeds']:
            def get(mode):
                return next(r for r in rows if r['mode'] == mode and r['seed'] == seed and
                            r['split'] == 'inner_holdout' and r['reset_steps'] == 275)['nh4']
            c, b = get(candidate), get(base)
            checks.append(dict(seed=seed, base=base, recall_not_worse=c['step_recall'] >= b['step_recall'],
                               fpr_not_worse=c['fpr'] <= b['fpr']))
            gains.append(b['fpr'] - c['fpr'])
        reductions[base] = float(np.mean(gains))
    passed = all(v for c in checks for k, v in c.items() if k not in ('seed', 'base')) and min(reductions.values()) >= .01
    return dict(candidate=candidate, checks=checks, mean_fpr_reduction=reductions, passed=bool(passed))


def main(a):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError(f'Refusing overwrite {a.out}')
    a.out.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(a.data)
    days = data['days']['train']
    assert len(days) == 330
    partition = dict(fit=days[:263], purge=days[263:264], inner_holdout=days[264:])
    xs = {s: arrays(dict(data=data['data'], days=partition), s) for s in ('fit', 'inner_holdout')}
    frame = routing_frame(a.raw)
    aux = {s: np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in partition[s]]) for s in xs}
    for s in xs:
        assert np.isfinite(aux[s]).all()
        for day in partition[s]:
            index = data['data'].loc[day].index
            assert len(index) == 288 and np.all(np.diff(index.asi8) == 300*10**9)
    state, exog = pack(xs['fit'])
    tensors = tuple(torch.tensor(z).float() for z in (state, exog, aux['fit']))
    scale = tensors[0].reshape(-1, 2).std(0, unbiased=False) + 1e-6
    bounds = torch.tensor(np.quantile(xs['fit']['temp'][:, 12:-1], [1/3, 2/3])).float()
    calibration, _ = day_batch(tensors, np.random.default_rng(CFG['calibration_seed']))
    prior_path = Path(__file__).parents[1] / 'e13/trial_summary.json'
    prior = json.loads(prior_path.read_text())
    inputs = {str(p): sha(p) for p in (a.data, a.raw)}
    assert inputs == prior['input_sha256']
    paths = [Path(__file__), Path(__file__).parents[1] / 'e13/experiment.py',
             Path(__file__).parents[1] / 'e12/trial.py', Path(__file__).parents[1] / 'e11/routing_trial.py',
             Path(__file__).parents[1] / 'e10/history_trial.py']
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), config=CFG, groups=GROUPS,
                  partition=partition, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  input_sha256=inputs, source_sha256={str(p.relative_to(Path(__file__).parents[3])): sha(p) for p in paths},
                  prior_anchor_config_sha256=sha(prior_path), state_scale=scale.tolist(), temperature_bounds=bounds.tolist(),
                  test_used=False, outer_val_used=False, internal_holdout_reused=True,
                  versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
                  calibration=[], training=[], evaluation=[], baseline_sha256={})
    for seed in CFG['seeds']:
        sources = dict(reference=a.e11 / f'with_routing_s{seed}.pt', single_mse=a.e12 / f'single_mse_s{seed}.pt',
                       start_multi=a.e12 / f'multi_mse_s{seed}.pt')
        for mode, path in sources.items():
            assert sha(path) == prior['baseline_sha256'][str(path)]
            report['baseline_sha256'][str(path)] = sha(path)
            report['evaluation'] += evaluate(load_model(path), xs, aux, mode, seed, a.out)
        source = sources['start_multi']
        model = load_model(source)
        t0 = time.monotonic()
        multi, anchor, (proxy, _, count) = objectives(model, *calibration, scale, bounds, 275)
        norm = lambda loss: float(torch.sqrt(sum(g.square().sum() for g in torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True))))
        gm, gp = norm(multi), norm((proxy * (count >= CFG['min_negative_count'])).mean())
        alpha = CFG['rate_gradient_ratio'] * gm / max(gp, 1e-12)
        beta = next(c['anchor_weight'] for c in prior['calibration'] if c['seed'] == seed)
        report['calibration'].append(dict(seed=seed, anchor_weight=beta, rate_weight=alpha,
                                          multi_grad_norm=gm, proxy_grad_norm=gp, elapsed_seconds=time.monotonic()-t0))
        print('calibration', report['calibration'][-1], flush=True)
        del multi, anchor, proxy, model
        for mode in MODES:
            model = load_model(source)
            opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
            rng = np.random.default_rng(CFG['sampling_seed_base'] + seed)
            digest = hashlib.sha256()
            multiplier, average = torch.zeros(7, 2), torch.zeros(7, 2)
            initialized = torch.zeros(7, 2, dtype=torch.bool)
            support = torch.zeros(7, 2, dtype=torch.int64)
            logs = []
            for update in tqdm(range(CFG['updates']), desc=f'{mode} seed {seed}', mininterval=5):
                batch, signature = day_batch(tensors, rng)
                digest.update(signature)
                multi, anchor, (proxy, fp, count) = objectives(model, *batch, scale, bounds, 36 if mode == 'short_mse' else 275)
                loss = multi + beta * anchor + alpha * penalty(proxy, multiplier, count, mode)
                opt.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'], error_if_nonfinite=True)
                assert torch.isfinite(loss)
                opt.step()
                valid = count >= CFG['min_negative_count']
                support += valid
                if mode == 'day_group':
                    multiplier, average, initialized = dual_step(multiplier, average, initialized, fp, count)
                elif mode == 'day_global':
                    m, av, init = dual_step(multiplier[:1], average[:1], initialized[:1], fp[:1], count[:1])
                    multiplier[:1], average[:1], initialized[:1] = m, av, init
                assert torch.isfinite(multiplier).all()
                logs.append(dict(update=update+1, multi=float(multi.detach()), anchor=float(anchor.detach()),
                                 fpr=(fp/count.clamp_min(1)).tolist(), negative_count=count.tolist(),
                                 multiplier=multiplier.tolist(), grad=float(grad)))
            path = a.out / f'{mode}_s{seed}.pt'
            torch.save(model.state_dict(), path)
            restored = load_model(path)
            probe = [v[:, 0] if i != 2 else v[:, 0].reshape(-1, 4, 2) for i, v in enumerate(calibration)]
            torch.testing.assert_close(model(*probe), restored(*probe), atol=0, rtol=0)
            report['training'].append(dict(mode=mode, seed=seed, batch_sha256=digest.hexdigest(), initial_sha256=sha(source),
                                           model_sha256=sha(path), first=logs[0], last=logs[-1], supported_updates=support.tolist()))
            write(a.out / f'{mode}_s{seed}_log.json', logs)
            report['evaluation'] += evaluate(restored, xs, aux, mode, seed, a.out)
            write(a.out / 'progress.json', report)
    for seed in CFG['seeds']:
        rows = [r for r in report['training'] if r['seed'] == seed]
        assert len({r['initial_sha256'] for r in rows}) == len({r['batch_sha256'] for r in rows}) == 1
    report['gates'] = [gates(report['evaluation'], 'inner_holdout', mode) for mode in MODES]
    report['tradeoff_gates'] = [tradeoff_gate(report['evaluation'], mode) for mode in ('day_global', 'day_group')]
    report['checks'] = dict(same_initial_weights=True, same_days_and_targets=True, finite_training=True,
                            restore_exact=True, future_state_not_used=True, false_negative_proxy_absent=True)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    write(a.out / 'results.json', report)
    print(json.dumps(report['tradeoff_gates']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('raw', 'data', 'e11', 'e12', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    main(p.parse_args())
