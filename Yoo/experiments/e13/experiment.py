"""연속 예측에 한 스텝 정확도 손실과 거짓경보 제약을 각각 추가한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e12'))
from trial import load_model, routing_frame, replay, arrays, pack, score, gates, write

MODES = ('multi', 'multi_anchor', 'multi_rate', 'multi_anchor_rate')
CFG = dict(horizon=36, updates=300, batch=128, lr=1e-4, weight_decay=1e-5,
           grad_clip=1., seeds=[0, 1], sampling_seed_base=13000,
           calibration_seed=13999, anchor_gradient_ratio=1., rate_gradient_ratio=.5,
           thresholds=[4., 2.], fpr_cap=[.1, .1], dual_lr=1., dual_ema=.9,
           dual_initial=1., dual_upper_bound=None, checkpoint='last',
           primary_candidate='multi_anchor_rate')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditional_terms(pred, target, temperature):
    """위반 및 비위반의 분모를 분리해 손실과 실제 FPR을 계산한다."""
    threshold = pred.new_tensor(CFG['thresholds'])
    positive = target > threshold
    negative = ~positive
    axes = tuple(range(pred.ndim - 1))
    npos, nneg = positive.sum(axes), negative.sum(axes)
    logits = (pred - threshold) / temperature
    fn = (F.softplus(-logits) * positive).sum(axes) / npos.clamp_min(1)
    fp = (F.softplus(logits) * negative).sum(axes) / nneg.clamp_min(1)
    hard = ((pred.detach() > threshold) & negative).sum(axes) / nneg.clamp_min(1)
    return fn, fp, hard, nneg > 0


def objectives(model, state, exog, extra, scale):
    pred = state[:, 0]
    sequence = []
    for t in range(exog.shape[1]):
        pred = model(pred, exog[:, t], extra[:, t].reshape(-1, 4, 2))
        sequence.append(pred)
    sequence = torch.stack(sequence, 1)
    multi = ((sequence - state[:, 1:]) / scale).square().mean()
    teacher = model(state[:, :-1].reshape(-1, 2), exog.reshape(-1, 6), extra.reshape(-1, 4, 2))
    anchor = ((teacher - state[:, 1:].reshape(-1, 2)) / scale).square().mean()
    return multi, anchor, conditional_terms(sequence, state[:, 1:], model.delta_sd)


def dual_step(multiplier, average, observed, valid):
    observed = observed.detach()
    new_average = observed if average is None else CFG['dual_ema'] * average + (1 - CFG['dual_ema']) * observed
    new_average = torch.where(valid, new_average, average if average is not None else observed)
    updated = (multiplier + CFG['dual_lr'] * (new_average - observed.new_tensor(CFG['fpr_cap'])) * valid).clamp_min(0)
    return updated.detach(), new_average.detach()


def window(tensors, rng):
    state, exog, aux = tensors
    days = rng.integers(0, len(state), CFG['batch'])
    starts = rng.integers(12, 288 - CFG['horizon'], CFG['batch'])
    times = torch.tensor(starts)[:, None] + torch.arange(CFG['horizon'] + 1)
    ids = torch.tensor(days)[:, None]
    return (state[ids, times], exog[ids, times[:, :-1]], aux[ids, times[:, :-1]]), days.tobytes() + starts.tobytes()


def calibrate(model, batch, scale):
    multi, anchor, (fn, fp, _, _) = objectives(model, *batch, scale)
    parameters = tuple(model.parameters())
    norm = lambda loss: float(torch.sqrt(sum(g.square().sum() for g in torch.autograd.grad(loss, parameters, retain_graph=True))))
    gm, ga, gr = norm(multi), norm(anchor), norm((fn + fp).mean())
    return dict(anchor_weight=CFG['anchor_gradient_ratio'] * gm / max(ga, 1e-12),
                rate_weight=CFG['rate_gradient_ratio'] * gm / max(gr, 1e-12),
                multi_norm=gm, anchor_norm=ga, rate_norm=gr)


def evaluate(model, xs, aux, mode, seed, out):
    rows = []
    for split, x in xs.items():
        for reset in (1, 36, 275):
            pred = replay(model, x, aux[split], 'with_routing', reset)
            row = dict(mode=mode, seed=seed, split=split, reset_steps=reset, **score(pred, x))
            if split == 'inner_holdout' and reset == 275:
                path = out / f'{mode}_s{seed}_holdout.npz'
                np.savez_compressed(path, pred=pred)
                row['prediction_sha256'] = sha(path)
            rows.append(row)
    tiny = {k: v[:2].copy() for k, v in xs['inner_holdout'].items()}
    expected = replay(model, tiny, aux['inner_holdout'][:2], 'with_routing', 275)
    tiny['nh4'][:, 13:] += 1000
    tiny['do'][:, 13:] += 1000
    np.testing.assert_array_equal(expected, replay(model, tiny, aux['inner_holdout'][:2], 'with_routing', 275))
    return rows


def tradeoff_gate(rows, candidate):
    checks = []
    gains = {}
    for base in ('start_multi', 'multi'):
        reductions = []
        for seed in CFG['seeds']:
            def get(mode):
                return next(r for r in rows if r['mode'] == mode and r['seed'] == seed
                            and r['split'] == 'inner_holdout' and r['reset_steps'] == 275)
            c, b = get(candidate), get(base)
            reductions.append(b['nh4']['fpr'] - c['nh4']['fpr'])
            checks.append(dict(base=base, seed=seed,
                               nh4_recall_not_worse=c['nh4']['step_recall'] >= b['nh4']['step_recall'],
                               nh4_fpr_not_worse=c['nh4']['fpr'] <= b['nh4']['fpr']))
        gains[base] = float(np.mean(reductions))
    passed = all(v for r in checks for k, v in r.items() if k not in ('base', 'seed')) and min(gains.values()) >= .01
    return dict(candidate=candidate, per_seed=checks, mean_fpr_reduction=gains, passed=bool(passed))


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
    for split in xs:
        assert np.isfinite(aux[split]).all()
        for day in partition[split]:
            index = data['data'].loc[day].index
            assert len(index) == 288 and np.all(np.diff(index.asi8) == 300 * 10**9)
    state, exog = pack(xs['fit'])
    tensors = tuple(torch.tensor(z).float() for z in (state, exog, aux['fit']))
    scale = tensors[0].reshape(-1, 2).std(0, unbiased=False) + 1e-6
    calibration, _ = window(tensors, np.random.default_rng(CFG['calibration_seed']))
    source_paths = [Path(__file__), Path(__file__).parents[1] / 'e12/trial.py',
                    Path(__file__).parents[1] / 'e11/routing_trial.py', Path(__file__).parents[1] / 'e10/history_trial.py']
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), config=CFG, partition=partition,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256={str(p.relative_to(Path(__file__).parents[3])): sha(p) for p in source_paths},
                  input_sha256={str(p): sha(p) for p in (a.data, a.raw)}, state_scale=scale.tolist(),
                  versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
                  test_used=False, outer_val_used=False, internal_holdout_reused=True,
                  training=[], calibration=[], evaluation=[], baseline_sha256={})
    for seed in CFG['seeds']:
        sources = dict(reference=a.e11 / f'with_routing_s{seed}.pt',
                       single_mse=a.e12 / f'single_mse_s{seed}.pt',
                       start_multi=a.e12 / f'multi_mse_s{seed}.pt')
        for mode, source in sources.items():
            report['baseline_sha256'][str(source)] = sha(source)
            report['evaluation'] += evaluate(load_model(source), xs, aux, mode, seed, a.out)
        source = sources['start_multi']
        factors = calibrate(load_model(source), calibration, scale)
        report['calibration'].append(dict(seed=seed, **factors))
        for mode in MODES:
            model = load_model(source)
            opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
            rng = np.random.default_rng(CFG['sampling_seed_base'] + seed)
            digest = hashlib.sha256()
            multiplier = torch.full((2,), CFG['dual_initial'])
            average = None
            logs = []
            for update in tqdm(range(CFG['updates']), desc=f'{mode} seed {seed}', mininterval=5):
                batch, signature = window(tensors, rng)
                digest.update(signature)
                multi, anchor, (fn, fp, hard, valid) = objectives(model, *batch, scale)
                loss = multi
                if 'anchor' in mode:
                    loss = loss + factors['anchor_weight'] * anchor
                if 'rate' in mode:
                    loss = loss + factors['rate_weight'] * (fn + multiplier * fp).mean()
                opt.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'], error_if_nonfinite=True)
                assert torch.isfinite(loss)
                opt.step()
                if 'rate' in mode:
                    multiplier, average = dual_step(multiplier, average, hard, valid)
                assert torch.isfinite(multiplier).all()
                logs.append(dict(update=update + 1, multi=float(multi.detach()), anchor=float(anchor.detach()),
                                 fn_proxy=fn.detach().tolist(), fp_proxy=fp.detach().tolist(),
                                 window_fpr=hard.tolist(), multiplier=multiplier.tolist(), grad=float(grad)))
            path = a.out / f'{mode}_s{seed}.pt'
            torch.save(model.state_dict(), path)
            restored = load_model(path)
            probe = [v[:, 0] if i != 2 else v[:, 0].reshape(-1, 4, 2) for i, v in enumerate(calibration)]
            torch.testing.assert_close(model(*probe), restored(*probe), atol=0, rtol=0)
            report['training'].append(dict(mode=mode, seed=seed, batch_sha256=digest.hexdigest(),
                                           initial_sha256=sha(source), model_sha256=sha(path), first=logs[0], last=logs[-1]))
            write(a.out / f'{mode}_s{seed}_log.json', logs)
            report['evaluation'] += evaluate(restored, xs, aux, mode, seed, a.out)
            write(a.out / 'progress.json', report)
    for seed in CFG['seeds']:
        rows = [r for r in report['training'] if r['seed'] == seed]
        assert len({r['initial_sha256'] for r in rows}) == len({r['batch_sha256'] for r in rows}) == 1
    report['gates'] = [gates(report['evaluation'], 'inner_holdout', mode) for mode in MODES]
    report['tradeoff_gates'] = [tradeoff_gate(report['evaluation'], mode) for mode in MODES if mode != 'multi']
    report['checks'] = dict(same_initial_weights=True, same_windows=True, finite_training=True,
                            restore_exact=True, future_state_not_used=True)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    write(a.out / 'results.json', report)
    print(json.dumps([dict(candidate=g['candidate'], screen=g['screen_passed'], eligible=g['pilot_eligibility']) for g in report['gates']]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('raw', 'data', 'e11', 'e12', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
