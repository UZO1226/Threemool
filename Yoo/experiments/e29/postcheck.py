"""완료한 Kaggle 시험의 출처, 불변 요소, 학습량과 보고 단위를 검사한다."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main(root, repo, out, prepared_root):
    manifest = json.loads((root / 'pilot/manifest.json').read_text())
    prepared = json.loads((prepared_root / 'preparation.json').read_text())
    for path, expected in prepared['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    for path, expected in {**manifest['inputs'], **manifest['reused_inputs']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    for path, expected in manifest['source'].items():
        assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == expected, path
    prior = torch.load(prepared_root / 'prior.pt', weights_only=True)
    normalizer = json.loads((prepared_root / 'normalizer.json').read_text())
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
        if m['method'] == 'fcsrl_preserve':
            for r in log:
                assert r['preserve_after_error'] <= r['preserve_before_error'] + 1e-10
                assert r['preserve_after_kl'] <= r['preserve_before_kl'] * 1.01 + 1e-10
                if r['preserve_accepted']:
                    assert r['preserve_after_error'] <= .8 * r['preserve_before_error'] + 1e-10
                assert r['preserve_fit_samples'] == 2048 and r['preserve_check_samples'] == 1024
        if m['method'] in ('fcsrl_actor', 'fcsrl_critic', 'fcsrl_extra_ppo'):
            assert config['preservation']['mode'] == m['method']
            for r in log:
                assert r['preserve_fit_samples'] == 2048 and r['preserve_check_samples'] == 1024
                assert r['extra_attempted_steps'] in (0, 50)
                if m['method'] == 'fcsrl_actor':
                    assert r['preserve_before_value_mse'] == r['preserve_after_value_mse']
                    assert r['preserve_after_kl'] <= r['preserve_before_kl'] * 1.01 + 1e-10
                elif m['method'] == 'fcsrl_critic':
                    assert r['preserve_before_kl'] == r['preserve_after_kl']
                    assert r['preserve_after_value_mse'] <= r['preserve_before_value_mse'] + 1e-10
                else:
                    assert r['extra_attempted_steps'] == 50
                    assert r['extra_anchor_kl'] <= config['config']['target_kl'] + 1e-7
        checks.append({'method': m['method'], 'seed': m['seed'], 'passed': True,
                       'max_lambda': max(r['lambda'] for r in log),
                       'last_behavior_penalty': log[-1]['behavior_penalty_mean'],
                       'last_std': log[-1]['std'],
                       'max_encoder_sync_kl': max(r.get('encoder_sync_kl', 0.) for r in log),
                       'accepted_syncs': sum(r.get('preserve_accepted', 0.) for r in log),
                       'before_kl_sum': sum(r.get('preserve_before_kl', 0.) for r in log),
                       'after_kl_sum': sum(r.get('preserve_after_kl', 0.) for r in log)})
    assert len(checks) == 25
    assert sum(r['method'] == 'fcsrl_preserve' for r in checks) == 5
    report = {'passed': True, 'input_and_training_source_hashes_unchanged': True,
              'prior_weights_and_normalizer_unchanged_all_runs': True,
              'last_checkpoint_mlp_evaluation_exact': True, 'two_model_val_only_215_unique_episodes': True,
              'reward_penalty_accounting': True, 'steps_per_run': 202048, 'runs': checks}
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'repo', 'out', 'prepared-root'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.repo, a.out, a.prepared_root)
