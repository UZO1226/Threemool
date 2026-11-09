"""완료한 Kaggle 시험의 출처, 불변 요소, 학습량과 보고 단위를 검사한다."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main(root, repo, out):
    manifest = json.loads((root / 'pilot/manifest.json').read_text())
    prepared = json.loads((root / 'prepared/preparation.json').read_text())
    for path, expected in prepared['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    for path, expected in manifest['inputs'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    for path, expected in manifest['source'].items():
        assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == expected, path
    prior = torch.load(root / 'prepared/prior.pt', weights_only=True)
    normalizer = json.loads((root / 'prepared/normalizer.json').read_text())
    checks = []
    for p in sorted((root / 'pilot/runs').glob('*/seed_*/metrics.json')):
        run = p.parent
        m = json.loads(p.read_text())
        config = json.loads((run / 'effective_config.json').read_text())
        log = json.loads((run / 'train_log.json').read_text())
        cross = json.loads((run / 'cross_evaluation.json').read_text())
        saved = torch.load(run / 'model_final/prior.pt', weights_only=True)
        assert all(torch.equal(saved[k], v) for k, v in prior.items())
        assert json.loads((run / 'model_final/normalizer.json').read_text()) == normalizer
        assert len(log) == 22 and m['steps'] == 202048
        assert len([r for r in log if 'val' in r]) == 2
        beta = config['behavior_beta']
        assert beta == (1. if m['method'].endswith('_anchor') else 0.)
        for r in log:
            assert np.isclose(r['shaped_reward_mean'], r['base_reward_mean'] - beta * r['behavior_penalty_mean'], atol=1e-6)
        for tag in ('mlp', 'gbm'):
            rs = cross[tag]['episodes']
            assert len(rs) == 215 and len({(r['day'], r['member']) for r in rs}) == 215
            assert set(r['day'] for r in rs) == set(config['validation']['days'])
            assert len(config['validation']['days']) == 43
            assert all('2024-01-01' <= r['day'] <= '2024-02-29' for r in rs)
        for key in ('air_m3', 'cost', 'reward', 'nh4_viol_pct', 'do_viol_pct'):
            assert abs(m['val_final'][key] - cross['mlp']['summary'][key]) < 1e-8
        checks.append({'method': m['method'], 'seed': m['seed'], 'passed': True,
                       'max_lambda': max(r['lambda'] for r in log),
                       'last_behavior_penalty': log[-1]['behavior_penalty_mean']})
    assert len(checks) == 8
    report = {'passed': True, 'input_and_training_source_hashes_unchanged': True,
              'prior_weights_and_normalizer_unchanged_all_runs': True,
              'last_checkpoint_mlp_evaluation_exact': True, 'two_model_val_only_215_unique_episodes': True,
              'reward_penalty_accounting': True, 'steps_per_run': 202048, 'runs': checks}
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'repo', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.repo, a.out)
