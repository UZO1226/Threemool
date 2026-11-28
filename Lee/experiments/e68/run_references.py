"""Build and independently authenticate every fresh e63 comparator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_baselines import ROOT, SEEDS, sha, write

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'baselines', 'rates', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rates = json.loads((args.rates / 'rate_preparation_summary.json').read_text())
    assert rates['passed'] and rates['policy_seeds'] == SEEDS
    assert json.loads((args.rates / 'study_status.json').read_text())['phase'] == 'complete'
    sources = {str(path.resolve()): sha(path) for path in HERE.glob('*.py')}
    manifest = json.loads((args.rates / 'study_manifest.json').read_text())
    sources.update(manifest['source_sha256'])
    args.out.mkdir(parents=True)
    write(args.out / 'study_manifest.json', {'source_sha256': sources, 'policy_seeds': SEEDS,
        'rate_summary_sha256': sha(args.rates / 'rate_preparation_summary.json'),
        'test_evaluated': False, 'candidate_evaluation': False, 'base_training_steps': 401280})
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1', DARROW_FAST_DYNAMICS='1', PYTHONIOENCODING='utf-8')
    completed = []
    def phase(name, jobs):
        pending, active = jobs.copy(), []
        try:
            while pending or active:
                assert all(sha(Path(path)) == expected for path, expected in sources.items())
                while pending and len(active) < 2:
                    label, command = pending.pop(0)
                    log = (args.out / f'{label}.log').open('w', encoding='utf-8')
                    process = subprocess.Popen([sys.executable, '-u', *map(str, command)],
                        cwd=args.workspace, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    active.append((label, process, log))
                    print('START', label, process.pid, flush=True)
                for label, process, log in active.copy():
                    code = process.poll()
                    if code is None:
                        continue
                    active.remove((label, process, log))
                    log.close()
                    if code:
                        raise RuntimeError(f'{label} exited{code}; inspect preserved log')
                    completed.append(label)
                    print('COMPLETE', label, flush=True)
                write(args.out / 'study_status.json', {'phase': name, 'completed': completed,
                    'running': [{'label': label, 'pid': process.pid} for label, process, _ in active],
                    'pending': [label for label, _ in pending]})
                if active:
                    time.sleep(1)
        except BaseException:
            for _, process, log in active:
                process.terminate()
                process.wait()
                log.close()
            write(args.out / 'failure.json', {'phase': name, 'terminal': True, 'completed': completed})
            raise
    common = ['--workspace', args.workspace.resolve(), '--baselines', args.baselines.resolve(),
              '--rates', args.rates.resolve()]
    phase('reference_construction', [(f'reference_s{seed}', [HERE / 'build_reference.py', *common,
          '--policy-seed', seed, '--out', args.out.resolve() / f'seed_{seed}']) for seed in SEEDS])
    phase('independent_audits', [(f'audit_s{seed}', [HERE / 'audit_reference.py', *common,
          '--reference', args.out.resolve() / f'seed_{seed}',
          '--out', args.out.resolve() / f'seed_{seed}/action_audit.json',
          *(['--null-check'] if seed == 5 else [])]) for seed in SEEDS])
    records = []
    for seed in SEEDS:
        folder = args.out / f'seed_{seed}'
        pair = json.loads((folder / 'summary.json').read_text())
        audit = json.loads((folder / 'action_audit.json').read_text())
        assert pair['policy_seed'] == audit['policy_seed'] == seed and audit['passed']
        assert audit['summary_sha256'] == sha(folder / 'summary.json')
        assert set(audit['replays']) == {'native', 'matched'}
        assert all(r['action_rows'] == 152000 and r['trajectory_replay_exact'] and
                   r['independent_sklearn_actions_exact'] and r['executed_air_accounting_exact']
                   for r in audit['replays'].values())
        records.append({'policy_seed': seed, 'base_checkpoint_hashes': pair['base_checkpoint_hashes'],
            'rate_model_sha256': pair['rate_model_sha256'],
            'controller_config_sha256': pair['controller_config_sha256'],
            'fixed_e63_offset_after_wrapper': pair['matching']['offset'],
            'primary_reference_metrics': pair['e63_matched'], 'e63_native_metrics': pair['e63_native'],
            'original_fcsrl_metrics': pair['original_fcsrl_native'],
            'original_fcsrl_air_match': pair['matching']['matched_within_one_percent'],
            'summary_sha256': sha(folder / 'summary.json'), 'action_audit_sha256': sha(folder / 'action_audit.json'),
            'matched_file_sha256': sha(folder / 'e63_matched.json'),
            'freeze_canonical_sha256': pair['freeze_canonical_sha256'], 'data_sha256': pair['data_sha256']})
    assert len({r['freeze_canonical_sha256'] for r in records}) == len({r['data_sha256'] for r in records}) == 1
    write(args.out / 'baseline_authentication.json', {'authentication_passed': True, 'policy_seeds': SEEDS,
        'records': records, 'all_original_fcsrl_air_matches': all(r['original_fcsrl_air_match'] for r in records),
        'test_evaluated': False, 'candidate_evaluation': False,
        'scope': 'Authenticates actual fresh e63 controllers and achieved air only, not a candidate improvement. Original-FCSRL air matches are reported separately.'})
    write(args.out / 'study_status.json', {'phase': 'complete', 'completed': completed, 'running': [], 'pending': []})
    print('FRESH_REFERENCE_STUDY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
