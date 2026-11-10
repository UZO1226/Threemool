"""e31 2단계 준비: 계절을 섞은 분할로 DARROW 전이 모델과 기록 운전 초기 정책을 새로 학습한다

e30 prepare.py 와 같은 절차(단조 GBM 5구성원, 기록 총 공기량 모방 초기 정책, 고정 정규화)를 쓰고 분할만 바꾼다.
- 평가(test): e30 과 같은 마지막 41일. 이 단계에서도 쓰지 않는다
- 나머지 228일을 날짜 순서의 7일 블록으로 나눠 블록 번호 % 5 == 2 인 블록을 검증, 나머지를 학습으로 쓴다.
  학습과 검증 모두에 5 ~ 12월이 들어가 계절 외삽과 알고리즘 차이가 덜 섞인다

    python prepare.py --raw <원자료 폴더> --out /home/jio/fcsrl_src/results/e31/prepared
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
from common import save, sha  # noqa: E402
from ppo_lag import RunningNorm, mlp  # noqa: E402
from prep_darrow import load, add_influent, complete_days, split_days  # noqa: E402
from surrogate_dw import train_gbm  # noqa: E402
from darrow_env import AIR_MAX, load_days  # noqa: E402


def block_split(days, test_n=41, block=7, val_every=5, val_pos=2):
    test = days[-test_n:]
    rest = days[:-test_n]
    train, val = [], []
    for i, d in enumerate(rest):
        (val if (i // block) % val_every == val_pos else train).append(d)
    return {'train': train, 'val': val, 'test': test}


def main(raw, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(3100)
    online = raw / 'train_val_test_online_dataset.csv'
    influent = raw / 'train_val_test_dynamic_influent.csv'
    table, alignment = add_influent(load(online), influent)
    days = complete_days(table)
    splits = block_split(days)
    e30 = split_days(days)
    assert splits['test'] == e30['test'], 'e30 과 같은 평가 구간이어야 한다'
    train_table = pd.concat([table.loc[d] for d in splits['train']])
    zones = [f'air{i}' for i in range(1, 6)]
    med = train_table[zones].median()
    pd.to_pickle({'data': table, 'days': splits, 'zone_fractions': (med / med.sum()).to_dict(),
                  'steps_per_day': 96}, out / 'data.pkl')
    months = lambda ds: pd.Series([d[:7] for d in ds]).value_counts().sort_index().to_dict()  # noqa: E731
    save(out / 'data_manifest.json', {
        'source': 'https://www.kaggle.com/competitions/dynamic-modeling-of-wastewater-treatment-process/data',
        'raw_sha256': {p.name: sha(p) for p in sorted(raw.iterdir()) if p.is_file()},
        'split_rule': '마지막 41일 test (e30 과 같음). 나머지는 7일 블록, 블록 번호 % 5 == 2 이면 val',
        'splits': splits, 'n': {k: len(v) for k, v in splits.items()},
        'months': {k: months(v) for k, v in splits.items() if k != 'test'}, 'alignment_correlation': alignment,
        'test_not_evaluated': True})
    print('split', {k: len(v) for k, v in splits.items()}, flush=True)
    log = train_gbm(table, splits, out / 'surrogate.pkl')
    save(out / 'surrogate_fit.json', log)
    _, day_data = load_days(out / 'data.pkl')
    fit = [d for i, d in enumerate(splits['train']) if i % 5 != 4]
    check = [d for i, d in enumerate(splits['train']) if i % 5 == 4]

    def arrays(ds):
        xs, ys = [], []
        for day in ds:
            x = day_data[day]
            prev = np.r_[x['air'][0], x['air'][:-1]]
            ex = x['exo']
            obs = np.column_stack([x['state'], ex[:, 0] / 1000, ex[:, 1] / 10, ex[:, 2], ex[:, 3], ex[:, 4] / 50,
                                   prev / AIR_MAX])
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
        bc_log.append({'epoch': epoch + 1, 'check_air_mae': float(abs(pred - yv).mean() * AIR_MAX / 2)})
    torch.save(net.state_dict(), out / 'prior.pt')
    save(out / 'normalizer.json', norm.state())
    save(out / 'preparation.json', {'fit_days': len(fit), 'check_days': len(check), 'seed': 3100, 'epochs': 40,
         'log': bc_log, 'input_sha256': {str(p): sha(p) for p in (online, influent, out / 'data.pkl', out / 'surrogate.pkl')}})
    print('PREPARED', bc_log[-1], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.raw, a.out)
