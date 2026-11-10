"""저장 배열에서 지표, 학습 정답, 고정 모델과 해시를 독립 검산한다."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out, source):
    manifest = json.loads((out / 'manifest.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    result = json.loads((out / 'comparison.json').read_text())
    fit, holdout, val = [protocol[k] for k in ('fit_dates', 'holdout_dates', 'validation_dates')]
    assert (len(fit), len(holdout), len(val)) == (140, 48, 40)
    assert max(fit) < min(holdout) and max(holdout) < min(val)
    assert not (set(fit) & set(holdout) or set(holdout) & set(val) or set(fit) & set(val))
    with (out / 'candidate.pkl').open('rb') as f:
        candidate = pickle.load(f)
    with (source / 'models.pkl').open('rb') as f:
        original = pickle.load(f)
    assert pickle.dumps(candidate['base']) == pickle.dumps(original['base'])
    assert pickle.dumps(candidate['nh4_models']['reference']) == pickle.dumps(original['history_nh4'])
    data = np.load(out / 'training_examples.npz')
    observed_x, observed_y = data['observed_x'], data['observed_y']
    generated_x, generated_y = data['generated_x'], data['generated_y']
    assert observed_x.shape == (13300, 44) and generated_x.shape == (13300, 5, 44)
    assert np.allclose(generated_x[..., 0] + generated_y, (observed_x[:, 0] + observed_y)[:, None], atol=1e-12, rtol=0)
    assert np.array_equal(generated_x[:, :, 5:24], np.repeat(observed_x[:, None, 5:24], 5, axis=1))
    # These recorded exogenous and airflow inputs are fixed even when water states differ.
    ids = np.load(out / 'training_row_ids.npy')
    for member in range(5):
        assert np.array_equal(ids[member], np.random.RandomState(member).choice(13300, 10640, replace=False))
        for arm in ('control', 'augmented'):
            model = candidate['nh4_models'][arm][member]
            ref = original['history_nh4'][member]
            assert model.n_iter_ == ref.n_iter_ and model.early_stopping is False
            params = ref.get_params()
            params.update(max_iter=ref.n_iter_, early_stopping=False)
            got = model.get_params()
            for key, value in params.items():
                assert np.array_equal(got[key], value)
            log = next(v for v in result['training'] if v['arm'] == arm and v['member'] == member)
            assert log['rows'] == 21280
            assert log['row_ids_sha256'] == hashlib.sha256(ids[member].tobytes()).hexdigest()
    z = np.load(out / 'predictions.npz')
    previous = np.load(source / 'predictions.npz')
    assert np.array_equal(z['reference_cold'], previous['history_pred'])
    assert np.array_equal(z['reference_holdout'], previous['history_calibration_pred'])
    assert np.array_equal(z['reference_events'], previous['history_events'])
    metric_checks = 0
    for name in ('reference', 'control', 'augmented'):
        for mode in ('holdout', 'cold', 'warm', 'one_step'):
            p = z[name + '_' + mode]
            a = z['one_step_actual'] if mode == 'one_step' else z['actual_' + mode]
            t = np.broadcast_to(a[:, None], p.shape)
            v = result['results'][name][mode]
            assert np.isfinite(p).all() and (p >= 0).all()
            assert (p <= candidate['base']['hi']).all()
            for j, state in enumerate(('nh4', 'no3', 'do1', 'do2', 'do3')):
                assert np.isclose(np.abs(t[..., j]-p[..., j]).mean(), v['mae'][state])
                assert np.isclose(np.sqrt(((t[..., j]-p[..., j])**2).mean()), v['rmse'][state])
                metric_checks += 2
            for target, truth, pred in (('NH4', t[..., 0] > 4, p[..., 0] > 4), ('DO', t[..., 2:].max(3) > 2, p[..., 2:].max(3) > 2)):
                tn, fp, fn, tp = confusion_matrix(truth.ravel(), pred.ravel(), labels=[False, True]).ravel()
                q = v['confusion'][target]
                assert [int(tp), int(fp), int(tp+fn), int(tn+fp)] == [q[k] for k in ('tp', 'fp', 'positives', 'negatives')]
                assert np.isclose(q['recall_pct'], 100*tp/(tp+fn))
                assert np.isclose(q['fpr_pct'], 100*fp/(fp+tn))
                errors = pred.sum(2)-truth.sum(2)
                assert np.isclose(np.abs(errors).mean(), v['counts'][target]['mae'])
                metric_checks += 3
            te = np.maximum(t[..., 0]-4, 0).sum(2)*.25
            pe = np.maximum(p[..., 0]-4, 0).sum(2)*.25
            assert np.isclose(np.abs(pe-te).mean(), v['nh4_excess_mg_h_L']['mae'])
            metric_checks += 1
        assert np.array_equal(z[name+'_one_step'][..., 1:], z['reference_one_step'][..., 1:])
        for e in result['events'][name]:
            step = e['minutes']//15
            a = z['event_actual']
            mask = z['event_dq'] > 0 if e['group'] == 'increase' else z['event_dq'] < 0
            if e['group'] == 'decrease_rising':
                mask &= a[:, step, 0]-a[:, 0, 0] > .1
            p = z[name+'_events'][mask, :, step, 0]
            truth = np.broadcast_to(a[mask, step, 0][:, None], p.shape)
            assert mask.sum() == e['windows']
            assert np.isclose(np.abs(p-truth).mean(), e['nh4_mae'])
            tn, fp, fn, tp = confusion_matrix((truth>4).ravel(), (p>4).ravel(), labels=[False, True]).ravel()
            assert [int(tp), int(fp), int(tp+fn), int(tn+fp)] == [e['confusion'][k] for k in ('tp', 'fp', 'positives', 'negatives')]
            metric_checks += 2
    assert all(sha(path) == value for path, value in manifest['input_sha256'].items())
    for filename, digest in manifest['source_sha256'].items():
        assert sha(Path(__file__).parent / filename) == digest
    for filename, key in (('candidate.pkl', 'candidate_sha256_before_evaluation'), ('protocol.json', 'protocol_sha256'), ('training_examples.npz', 'training_examples_sha256')):
        assert sha(out / filename) == manifest[key]
    check = {'passed': True, 'independent_metric_checks': metric_checks,
             'source_models_fixed': True, 'matched_training_settings': True,
             'synthetic_residual_targets_and_recorded_inputs': True,
             'disjoint_dates': True, 'frozen_hashes_unchanged': True}
    (out / 'verification.json').write_text(json.dumps(check, indent=2))
    print(check)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    main(args.out, args.source)
