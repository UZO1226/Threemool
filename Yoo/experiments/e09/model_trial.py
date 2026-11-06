"""같은 대리 모델의 한 스텝 추가 학습과 12스텝 연속 예측 학습을 대조한다."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from audit import arrays, simulate, Surrogate, write


class Delta(nn.Module):
    def __init__(self, params, member):
        super().__init__()
        for name in ('x_mu', 'x_sd', 'y_mu', 'y_sd'):
            self.register_buffer(name, torch.tensor(params[name], dtype=torch.float64))
        self.layers = nn.ModuleList()
        for w, b in params['members'][member]:
            w = torch.tensor(w, dtype=torch.float64)
            layer = nn.Linear(w.shape[0], w.shape[1], dtype=torch.float64)
            with torch.no_grad():
                layer.weight.copy_(w.T)
                layer.bias.copy_(torch.tensor(b, dtype=torch.float64))
            self.layers.append(layer)

    def forward(self, x):
        x = (x - self.x_mu) / self.x_sd
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x[:, 0] * self.y_sd + self.y_mu


class Coupled(nn.Module):
    def __init__(self, params, member):
        super().__init__()
        self.do = Delta(params['do'], member)
        self.nh4 = Delta(params['nh4'], member)

    def forward(self, state, exog):
        n, d = state[:, 0], state[:, 1]
        air, phase, temp, flow, s, c = exog.unbind(-1)
        ph = torch.stack([(phase == k).to(state.dtype) for k in (0, 1, 2)], -1)
        xd = torch.cat((d[:, None], air[:, None] / 1000, ph, temp[:, None],
                        n[:, None], flow[:, None] / 1000, s[:, None], c[:, None]), -1)
        dn = (d + self.do(xd)).clamp_min(0)
        xn = torch.cat((n[:, None], d[:, None], dn[:, None], ph, temp[:, None],
                        flow[:, None] / 1000, s[:, None], c[:, None]), -1)
        nn_ = (n + self.nh4(xn)).clamp_min(0)
        return torch.stack((nn_, dn), -1)


def pack(x):
    n, t = x['nh4'].shape
    angle = np.broadcast_to(2 * np.pi * np.arange(t) / 288, (n, t))
    state = np.stack((x['nh4'], x['do']), -1)
    exog = np.stack((x['air'], x['phase'], x['temp'], x['q_in'],
                     np.sin(angle), np.cos(angle)), -1)
    return state, exog


def export(params, models):
    out = copy.deepcopy(params)
    for kind in ('do', 'nh4'):
        out[kind].pop('val_mse_norm', None)  # 기존 모델의 점수를 새 모델 점수로 남기지 않는다.
        out[kind]['members'] = [[(l.weight.detach().cpu().double().numpy().T.tolist(),
                                  l.bias.detach().cpu().double().numpy().tolist())
                                 for l in getattr(model, kind).layers] for model in models]
    return out


def preflight(params, path, x):
    state, exog = pack(x)
    sur = Surrogate(path)
    for member in range(sur.n_members):
        model = Coupled(params, member)
        pred = torch.tensor(state[:, 0])
        seq = []
        with torch.no_grad():
            for t in range(12):
                pred = model(pred, torch.tensor(exog[:, t]))
                seq.append(pred.numpy())
        expected = simulate(sur, {k: v[:, :13] for k, v in x.items()}, member)
        np.testing.assert_allclose(np.stack(seq, 1), expected, atol=1e-9, rtol=1e-9)
    model = Coupled(params, 0).float()
    prediction = model(torch.tensor(state[:8, 0]).float(), torch.tensor(exog[:8, 0]).float())
    (prediction - torch.tensor(state[:8, 1]).float()).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS: NumPy/PyTorch 12-step parity for all members; finite joint gradients', flush=True)


def metrics(pred, truth):
    error = pred - truth
    cost = (pred[:, :, 0] > 4).astype(float) + (pred[:, :, 1] > 2)
    true_cost = (truth[:, :, 0] > 4).astype(float) + (truth[:, :, 1] > 2)
    return {'days': len(pred), 'nh4_mae': float(abs(error[:, :, 0]).mean()),
            'nh4_bias': float(error[:, :, 0].mean()), 'do_mae': float(abs(error[:, :, 1]).mean()),
            'predicted_cost': float(cost.sum(1).mean()), 'actual_cost': float(true_cost.sum(1).mean()),
            'daily_cost_mae': float(abs(cost.sum(1) - true_cost.sum(1)).mean())}


def evaluate(path, xs, floor):
    sur = Surrogate(path)
    rows = []
    for split, x in xs.items():
        truth = np.stack((x['nh4'][:, 1:], x['do'][:, 1:]), -1)
        cold = x['temp'].min(1) < floor
        for h in (1, 12, 36, 287):
            for member in range(sur.n_members):
                pred = simulate(sur, x, member, reset=h)
                for group, mask in [('all', np.ones(len(cold), bool)), ('cold', cold), ('in_range', ~cold)]:
                    if mask.any():
                        rows.append({'split': split, 'reset_steps': h, 'member': member, 'group': group,
                                     **metrics(pred[mask], truth[mask])})
    return rows


def main(a):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = pd.read_pickle(a.data)
    for split in ('train', 'val'):
        for day in data['days'][split]:
            index = data['data'].loc[day].index
            assert len(index) == 288 and index[0].hour == 0 and index[0].minute == 0
            assert np.all(np.diff(index.asi8) == 300 * 10**9), day
    xs = {s: arrays(data, s) for s in ('train', 'val')}
    params = json.loads(a.model.read_text())
    preflight(params, a.model, xs['train'])
    if a.preflight_only:
        return
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError(f'Refusing to overwrite {a.out}')
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = {'horizon': 12, 'updates': 300, 'batch_size': 128, 'lr': 1e-4, 'weight_decay': 1e-5,
           'grad_clip': 1., 'sampling_seed_base': 9000, 'checkpoint': 'last',
           'loss': 'mean squared state error divided by train state standard deviation squared'}
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'config': cfg,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.data, a.model)},
              'test_used': False, 'validation_used_for_gradients_or_checkpoint_selection': False,
              'pretrained_model_used_validation': True, 'device': a.device,
              'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
              'training': [], 'evaluation': {}}
    state, exog = pack(xs['train'])
    scale = state.reshape(-1, 2).std(0) + 1e-6
    report['state_loss_scale'] = scale.tolist()
    states = torch.tensor(state, dtype=torch.float32, device=a.device)
    exogs = torch.tensor(exog, dtype=torch.float32, device=a.device)
    scale = torch.tensor(scale, dtype=torch.float32, device=a.device)
    offset = torch.arange(cfg['horizon'] + 1, device=a.device)
    paths = {'original': a.model}
    for mode in ('one_step_joint', 'multi_step12'):
        models = []
        for member in range(len(params['do']['members'])):
            model = Coupled(params, member).float().to(a.device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            rng = np.random.default_rng(cfg['sampling_seed_base'] + member)
            losses = []
            batch_digest = hashlib.sha256()
            for update in tqdm(range(cfg['updates']), desc=f'{mode} member {member}', mininterval=5):
                day = rng.integers(0, len(state), cfg['batch_size'])
                start = rng.integers(0, state.shape[1] - cfg['horizon'], cfg['batch_size'])
                batch_digest.update(day.tobytes() + start.tobytes())
                ix = torch.tensor(start, device=a.device)[:, None] + offset
                iy = torch.tensor(day, device=a.device)[:, None]
                target, inputs = states[iy, ix], exogs[iy, ix[:, :-1]]
                prediction, loss = target[:, 0], 0
                for t in range(cfg['horizon']):
                    if mode == 'one_step_joint':
                        prediction = target[:, t]
                    prediction = model(prediction, inputs[:, t])
                    loss = loss + ((prediction - target[:, t + 1]) / scale).square().mean() / cfg['horizon']
                opt.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                opt.step()
                losses.append({'update': update + 1, 'loss': loss.item(), 'gradient_norm': grad.item()})
            report['training'].append({'mode': mode, 'member': member, 'batch_sha256': batch_digest.hexdigest(),
                                       'first_loss': losses[0]['loss'], 'last_loss': losses[-1]['loss']})
            write(a.out / f'{mode}_member{member}_log.json', losses)
            models.append(model.cpu())
            write(a.out / 'progress.json', report)
        candidate = export(params, models)
        candidate['e09_training'] = {'mode': mode, **cfg, 'state_loss_scale': report['state_loss_scale']}
        paths[mode] = a.out / f'{mode}.json'
        write(paths[mode], candidate)
        # 저장된 numpy 추론과 실제 학습망의 한 스텝 출력을 대조한다.
        saved = Surrogate(paths[mode])
        for m, model in enumerate(models):
            with torch.no_grad():
                p = model(torch.tensor(state[:, 0]).float(), torch.tensor(exog[:, 0]).float()).numpy()
            q = simulate(saved, {k: v[:, :2] for k, v in xs['train'].items()}, m)[:, 0]
            np.testing.assert_allclose(p, q, atol=2e-5, rtol=2e-5)
    for m in range(5):
        assert report['training'][m]['batch_sha256'] == report['training'][m + 5]['batch_sha256']
    report['checks'] = {'identical_paired_batches': True, 'export_inference_parity': True,
                        'finite_losses_and_gradients': True, 'within_day_regular_windows': True}
    for mode, path in paths.items():
        report['evaluation'][mode] = evaluate(path, xs, float(xs['train']['temp'].min()))
        write(a.out / 'results.json', report)
        print('EVALUATED', mode, flush=True)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    report['output_model_sha256'] = {mode: hashlib.sha256(path.read_bytes()).hexdigest() for mode, path in paths.items()}
    write(a.out / 'results.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--preflight-only', action='store_true')
    main(p.parse_args())
