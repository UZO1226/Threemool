"""평균 제약이 학습에서 느슨한 이유를 온도 및 운전 단계별로 진단한다."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from summarize import rates
from experiment import arrays, load_model, replay, routing_frame, score, sha, write


def main(a):
    torch.set_num_threads(1)
    if a.out.exists():
        raise ValueError(f'Refusing overwrite {a.out}')
    report = json.loads((a.trial / 'results.json').read_text())
    assert sha(a.data) == report['input_sha256'][str(a.data)]
    assert sha(a.raw) == report['input_sha256'][str(a.raw)]
    data = pd.read_pickle(a.data)
    xs = {s: arrays(dict(data=data['data'], days=report['partition']), s) for s in ('fit', 'inner_holdout')}
    bounds = np.quantile(xs['fit']['temp'][:, 12:-1], [1/3, 2/3])
    frame = routing_frame(a.raw)
    aux = np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in report['partition']['fit']])
    result = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  training_commit=report['commit'], training_performed=False, outer_val_used=False,
                  test_used=False, posthoc_descriptive=True, temperature_bounds=bounds.tolist(),
                  input_sha256=report['input_sha256'], rows=[], decomposition=[])
    for seed in (0, 1):
        for mode in ('start_multi', 'multi', 'multi_anchor_rate'):
            if mode == 'start_multi':
                source = Path(next(p for p in report['baseline_sha256'] if p.endswith(f'/multi_mse_s{seed}.pt')))
                expected = report['baseline_sha256'][str(source)]
            else:
                source = a.trial / f'{mode}_s{seed}.pt'
                expected = next(r['model_sha256'] for r in report['training'] if r['mode'] == mode and r['seed'] == seed)
            assert sha(source) == expected
            fit_pred = replay(load_model(source), xs['fit'], aux, 'with_routing', 275)
            original = next(r for r in report['evaluation'] if r['mode'] == mode and r['seed'] == seed and r['split'] == 'fit' and r['reset_steps'] == 275)
            assert all(original[k] == v for k, v in score(fit_pred, xs['fit']).items())
            path = a.trial / f'{mode}_s{seed}_holdout.npz'
            held = next(r for r in report['evaluation'] if r['mode'] == mode and r['seed'] == seed and r['split'] == 'inner_holdout' and r['reset_steps'] == 275)
            assert sha(path) == held['prediction_sha256']
            with np.load(path) as z:
                held_pred = z['pred']
            tables = {}
            for split, pred in [('fit', fit_pred), ('inner_holdout', held_pred)]:
                x = xs[split]
                temp, phase = x['temp'][:, 12:-1], x['phase'][:, 12:-1]
                bands = np.digitize(temp, bounds)
                groups = {'all': np.ones(temp.shape, dtype=bool)}
                groups.update({f'temp_{i}': bands == i for i in range(3)})
                groups.update({f'phase_{j}': phase == j for j in range(3)})
                groups.update({f'joint_{i}_{j}': (bands == i) & (phase == j) for i in range(3) for j in range(3)})
                tables[split] = {group: rates(pred[:, :, 0], x['nh4'][:, 13:], mask, 4) for group, mask in groups.items()}
                result['rows'] += [dict(mode=mode, seed=seed, split=split, group=group, **metrics) for group, metrics in tables[split].items()]
            train, held = tables['fit'], tables['inner_holdout']
            nneg = held['all']['fp'] + held['all']['tn']
            weighted = 0.
            for group in (f'joint_{i}_{j}' for i in range(3) for j in range(3)):
                if not held[group]['count']: continue
                weight = (held[group]['fp'] + held[group]['tn']) / nneg
                if weight:
                    assert train[group]['count'] and train[group]['fpr'] is not None
                    weighted += weight * train[group]['fpr']
            result['decomposition'].append(dict(mode=mode, seed=seed, fit_fpr=train['all']['fpr'],
                                                holdout_fpr=held['all']['fpr'], reweighted_fit_fpr=weighted,
                                                composition_difference=weighted-train['all']['fpr'],
                                                within_bin_difference=held['all']['fpr']-weighted))
    result['completed_at'] = datetime.now(timezone.utc).isoformat()
    write(a.out, result)
    print(json.dumps(result['decomposition'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('raw', 'data', 'trial', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
