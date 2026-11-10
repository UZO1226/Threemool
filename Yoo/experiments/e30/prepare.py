"""새 원본을 분할하고 DARROW 전이 모델과 공통 초기 운전 정책을 새로 학습한다."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from common import save, sha
from ppo_lag import RunningNorm, mlp
from prep_darrow import load, add_influent, complete_days, split_days
from surrogate_dw import train_gbm, build_arrays, STATE
from darrow_env import AIR_MAX, Dynamics, BatchEnv, load_days, all_episodes


def main(raw, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(2300)
    online = raw / 'train_val_test_online_dataset.csv'
    influent = raw / 'train_val_test_dynamic_influent.csv'
    table, alignment = add_influent(load(online), influent)
    splits = split_days(complete_days(table))
    assert [len(splits[k]) for k in ('train', 'val', 'test')] == [188, 40, 41]
    assert max(splits['train']) < min(splits['val']) < min(splits['test'])
    train_table = pd.concat([table.loc[d] for d in splits['train']])
    zones = [f'air{i}' for i in range(1, 6)]
    med = train_table[zones].median()
    pd.to_pickle({'data': table, 'days': splits, 'zone_fractions': (med / med.sum()).to_dict(),
                  'steps_per_day': 96}, out / 'data.pkl')
    save(out / 'data_manifest.json', {'source': 'https://www.kaggle.com/competitions/dynamic-modeling-of-wastewater-treatment-process/data',
         'raw_sha256': {p.name: sha(p) for p in sorted(raw.iterdir()) if p.is_file()},
         'used_for_training': [online.name, influent.name], 'splits': splits, 'alignment_correlation': alignment,
         'test_not_evaluated_by_e30': True, 'evaluation_files_have_no_online_quality_labels': True})
    print('TRAIN surrogate: 25 GBMs, training dates only', flush=True)
    log = train_gbm(table, splits, out / 'surrogate.pkl')
    save(out / 'surrogate_fit.json', log)
    _, day_data = load_days(out / 'data.pkl')
    # Fit/check both lie inside train. No old model, normalizer or prior is loaded.
    fit, check = splits['train'][:150], splits['train'][150:]

    def arrays(days):
        xs, ys = [], []
        for day in days:
            x = day_data[day]
            prev = np.r_[x['air'][0], x['air'][:-1]]
            ex = x['exo']
            obs = np.column_stack([x['state'], ex[:, 0] / 1000, ex[:, 1] / 10,
                                   ex[:, 2], ex[:, 3], ex[:, 4] / 50, prev / AIR_MAX])
            xs.append(obs[:-1])
            ys.append(np.clip(x['air'][:-1] / AIR_MAX * 2 - 1, -1, 1))
        return np.concatenate(xs), np.concatenate(ys)

    x, y = arrays(fit)
    xv, yv = arrays(check)
    norm = RunningNorm(11)
    norm.update(x)
    norm.m2 = np.maximum(norm.m2, norm.n * .05 ** 2)
    tx, ty = torch.tensor(norm(x), dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)
    vx = torch.tensor(norm(xv), dtype=torch.float32)
    net = mlp(11, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bc_log = []
    for epoch in range(40):
        for ids in torch.randperm(len(tx)).split(1024):
            loss = ((net(tx[ids]) - ty[ids]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = net(vx)[:, 0].clamp(-1, 1).numpy()
        bc_log.append({'epoch': epoch + 1, 'check_air_mae': float(abs(pred-yv).mean() * AIR_MAX / 2)})
    torch.save(net.state_dict(), out / 'prior.pt')
    save(out / 'normalizer.json', norm.state())
    save(out / 'preparation.json', {'fit_days': fit, 'check_days': check, 'seed': 2300, 'epochs': 40,
         'log': bc_log, 'input_sha256': {str(p): sha(p) for p in (online, influent, out / 'data.pkl', out / 'surrogate.pkl')}})
    print('PREPARED', bc_log[-1], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.raw, a.out)
