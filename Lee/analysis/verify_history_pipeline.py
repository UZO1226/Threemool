"""Verify completed e54 smoke/replay/matching artifacts before full seed training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    unit = json.loads(args.preflight.read_text())
    assert unit['passed'] and unit['same_capacity_exact']
    for name, expected in unit['runtime_sources'].items():
        payload = args.preflight.with_name(name).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(payload).hexdigest() == expected
    manifest = json.loads((args.runs / 'manifest.json').read_text())
    assert manifest['actual_steps_per_job'] == 6080 and len(manifest['jobs']) == 3
    assert json.loads((args.runs / 'status.json').read_text())['phase'] == 'complete'
    configs, natives, costs, result = {}, {}, {}, {}
    for name, method in [('baseline', 'ppolag_fcsrl'), ('history', 'ppolag_fcsrl_history'), ('repeated', 'ppolag_fcsrl_repeated')]:
        folder = args.runs / ('new_' + method) / 'seed_0'
        config = json.loads((folder / 'effective_config.json').read_text())
        metrics = json.loads((folder / 'metrics.json').read_text())
        assert json.loads((folder / 'progress.json').read_text())['phase'] == 'complete'
        assert config['seed'] == 0 and config['method'] == method and metrics['steps'] == 6080
        assert not config['experiment_metadata']['test_evaluated']
        native = json.loads((args.runs / ('native_' + name + '.json')).read_text())
        assert native['checkpoint_loaded_and_evaluated'] and native['smoke_only'] and not native['test_evaluated']
        assert native['method'] == method and native['policy_seed'] == 0 and native['training_steps'] == 6080
        assert native['summary']['n_episodes'] == 1600
        for key, value in metrics['val_final'].items():
            np.testing.assert_allclose(native['summary'][key], value, rtol=1e-10, atol=1e-10)
        checkpoint = folder / 'model_final'
        hashes = {str(path.relative_to(checkpoint)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(checkpoint.rglob('*')) if path.is_file()}
        assert hashes == native['checkpoint_hashes']
        log = json.loads((folder / 'train_log.json').read_text())
        assert len(log) == 1 and log[0]['steps'] == 6080
        costs[name] = log[0]['train_episode_cost']
        model = torch.load(checkpoint / 'model.pt', weights_only=True)
        width = 75 if name == 'baseline' else 81
        assert model['pi']['0.weight'].shape[1] == width
        normalizer = json.loads((checkpoint / 'normalizer.json').read_text())
        assert len(normalizer['mean']) == (11 if name == 'baseline' else 17)
        row = {'native_replay_passed': True, 'checkpoint_hashes_exact': True, 'steps': 6080}
        if name != 'baseline':
            extra = float(model['pi']['0.weight'][:, 11:17].abs().max())
            representation = torch.load(checkpoint / 'representation.pt', weights_only=True)
            encoder_extra = float(representation['encoder']['zs_net.layers.0.weight'][:, 11:].abs().max())
            assert extra > 0 and encoder_extra > 0
            row.update({'new_actor_columns_learned_max': extra, 'new_encoder_columns_learned_max': encoder_extra})
        configs[name], natives[name], result[name] = config, native, row
    assert configs['baseline']['config'] == configs['history']['config'] == configs['repeated']['config']
    assert configs['baseline']['representation'] == configs['history']['representation'] == configs['repeated']['representation']
    assert len(set(costs.values())) == 1
    for name in ('history', 'repeated'):
        pair = json.loads((args.runs / ('match_' + name) / 'summary.json').read_text())
        assert pair['smoke_only'] and pair['training_steps'] == 6080 and not pair['test_evaluated']
        assert pair['checkpoint_replay_verified'] and pair['baseline_method'] == 'ppolag_fcsrl'
        assert pair['baseline_checkpoint_hashes'] == natives['baseline']['checkpoint_hashes']
        assert pair['candidate_checkpoint_hashes'] == natives[name]['checkpoint_hashes']
        assert pair['matching']['matched_within_one_percent'] and pair['matching']['callback_queries'] > 0
        target, actual = pair['baseline_native']['air_m3'], pair['candidate_matched']['air_m3']
        assert abs(actual - target) <= target * .01
        result[name]['air_matching_smoke_passed'] = True
        result[name]['air_only_callback_queries'] = pair['matching']['callback_queries']
        result[name]['relative_air_error'] = pair['matching']['relative_air_error']
    output = {'passed': True, 'smoke_only': True, 'test_evaluated': False, 'full_policy_seeds_completed': [],
              'initial_collector_mean_cost_equal': costs, 'variants': result,
              'scope': 'Pipeline checks only. One6080-step smoke per method is not an extension performance result.'}
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
