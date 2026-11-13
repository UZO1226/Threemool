"""고정된 FCSRL 표현에서 현재 NH4와 다음 한 시간 초과 적분을 읽는 진단."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'e30'))
import common
from darrow_env import BatchEnv
sys.path.insert(0, str(HERE.parent / 'e45'))
from evaluate import load_selected, fingerprint
from fcsrl.utils import DeviceConfig


def windows(nh4):
    assert nh4.shape[-1] == 96
    excess = np.maximum(nh4 - 4, 0)
    return sum(excess[..., k:k + 92] for k in range(1, 5)) * .25


def dataset(dates, days):
    obs, act, now, future = [], [], [], []
    for day in dates:
        d = days[day]
        s, e, a = d['state'], d['exo'], d['air']
        previous = np.r_[a[0], a[:-1]]
        o = np.column_stack([s, e[:, 0] / 1000, e[:, 1] / 10,
                             e[:, 2], e[:, 3], e[:, 4] / 50, previous / 9000]).astype(np.float32)
        # 최초 상태의 관측 조립은 실제 환경 코드와 비교한다. 전이는 실행하지 않는다.
        env = BatchEnv({day: d}, None)
        np.testing.assert_array_equal(o[0], env.reset([(day, 0)])[0])
        obs.append(o[:92]); act.append(np.clip(a[:92] / 4500 - 1, -1, 1))
        now.append(s[:92, 0]); future.append(windows(s[:, 0]))
    return dict(obs=np.concatenate(obs), action=np.concatenate(act)[:, None],
                current_nh4=np.concatenate(now), future_excess=np.concatenate(future),
                dates=list(dates))


def scores(y, pred, dates):
    assert y.shape == pred.shape == (len(dates) * 92,)
    assert np.isfinite(pred).all()
    return dict(mae=float(mean_absolute_error(y, pred)), r2=float(r2_score(y, pred)),
                daily_mae=np.abs(y - pred).reshape(-1, 92).mean(1).tolist())


@torch.no_grad()
def features(agent, initial, d):
    norm = agent.obs_normalizer(d['obs'])
    ot = agent._t(norm)
    z = agent.fixed_encoder.zs(ot).numpy()
    zi = initial.zs(ot).numpy()
    pi = agent.policy_input(ot).numpy()
    np.testing.assert_array_equal(pi[:, :11], ot.numpy())
    np.testing.assert_array_equal(pi[:, 11:], z)
    n = agent.obs_normalizer
    raw_norm = (d['obs'] - n.mean) / np.sqrt(n.m2 / n.n + 1e-8)
    reconstructed = norm[:, 0] * np.sqrt(n.m2[0] / n.n + 1e-8) + n.mean[0]
    raw = np.column_stack([norm, d['action']])
    x = dict(raw=raw, trained_z=np.column_stack([z, d['action']]),
             initial_z=np.column_stack([zi, d['action']]),
             raw_plus_trained_z=np.column_stack([raw, z]),
             raw_plus_initial_z=np.column_stack([raw, zi]))
    diag = dict(clipped_feature_fraction=float((np.abs(raw_norm) > 10).mean()),
                nh4_clipped_count=int((np.abs(raw_norm[:, 0]) > 10).sum()),
                nh4_inverse_max_error=float(np.max(np.abs(reconstructed - d['current_nh4']))),
                policy_input_raw_prefix_exact=True)
    return x, diag


def run(out):
    assert sklearn.__version__ == '1.9.1'
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads((HERE / 'protocol.json').read_text())
    prepared = Path(p['prepared'])
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    assert torch.get_num_threads() == 1
    toy = np.arange(96, dtype=float)
    np.testing.assert_allclose(windows(toy), [np.maximum(toy[t+1:t+5]-4, 0).sum()*.25 for t in range(92)])
    np.testing.assert_array_equal(windows(np.full((2, 96), 4.)), np.zeros((2, 92)))
    split, days = load_selected(prepared / 'data.pkl')
    train = sorted(split['train'])
    groups = dict(fit=train[:112], tune=train[112:150], holdout=train[150:], val=sorted(split['val']))
    assert [len(v) for v in groups.values()] == [112, 38, 38, 40]
    assert len(set(sum(groups.values(), []))) == 228
    data = {k: dataset(v, days) for k, v in groups.items()}
    paths = [HERE / 'protocol.json', prepared / 'data.pkl', prepared / 'normalizer.json', prepared / 'prior.pt']
    for seed in p['seeds']:
        paths += sorted((Path(p['checkpoints']) / f'seed_{seed}' / 'model_final').glob('*'))
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file() and (str(ROOT) in path or '/fcsrl_src/fcsrl/' in path):
            paths.append(Path(path))
    paths += [Path(__file__)]
    hashes = {str(f): common.sha(f) for f in set(paths) if f.is_file()}
    manifest = dict(protocol=p, dates=groups, input_and_source_sha256=hashes,
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    started_at=datetime.now(timezone.utc).isoformat(), torch=torch.__version__,
                    sklearn=sklearn.__version__, cpu_threads=torch.get_num_threads(), test_evaluated=False)
    common.save(out / 'manifest.json', manifest)
    start = time.monotonic()
    results = []
    for seed in p['seeds']:
        agent = common.make_agent('fcsrl', seed, prepared, common.config())
        initial = deepcopy(agent.fixed_encoder)
        agent.load(Path(p['checkpoints']) / f'seed_{seed}' / 'model_final')
        before = fingerprint(agent)
        pairs = {k: features(agent, initial, d) for k, d in data.items()}
        feat = {k: v[0] for k, v in pairs.items()}
        result = dict(seed=seed, normalization={k: v[1] for k, v in pairs.items()}, targets={})
        for target in ['current_nh4', 'future_excess']:
            y = {k: d[target] for k, d in data.items()}
            records = {}
            for name in feat['fit']:
                candidates = []
                for alpha in p['ridge']['alphas']:
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='svd'))
                    model.fit(feat['fit'][name], y['fit'])
                    err = mean_absolute_error(y['tune'], model.predict(feat['tune'][name]))
                    candidates.append((float(err), alpha, model))
                _, alpha, model = min(candidates, key=lambda q: q[0])
                records[name] = dict(alpha=alpha, tune_maes={str(a): e for e, a, _ in candidates},
                    scores={k: scores(y[k], model.predict(feat[k][name]), groups[k]) for k in groups})
            model = HistGradientBoostingRegressor(**p['hgb']).fit(feat['fit']['raw'], y['fit'])
            records['raw_nonlinear'] = dict(scores={k: scores(y[k], model.predict(feat[k]['raw']), groups[k]) for k in groups})
            baseline = {k: d['current_nh4'] if target == 'current_nh4' else np.maximum(d['current_nh4'] - 4, 0) for k, d in data.items()}
            records['identity_or_persistence'] = dict(scores={k: scores(y[k], baseline[k], groups[k]) for k in groups})
            result['targets'][target] = records
        assert before == fingerprint(agent)
        result['policy_weights_unchanged'] = True
        results.append(result)
        common.save(out / f'seed_{seed}.json', result)
        common.save(out / 'status.json', dict(phase='running', seeds_completed=seed + 1, total=5, elapsed_seconds=time.monotonic()-start))
        print(f'SEED {seed} COMPLETE {time.monotonic()-start:.1f}s', flush=True)
    assert all(common.sha(path) == expected for path, expected in hashes.items())
    descriptive = []
    for r in results:
        m = {k: v['scores']['holdout']['mae'] for k, v in r['targets']['future_excess'].items()}
        descriptive.append(m['raw_plus_trained_z'] <= .9 * min(m['raw'], m['raw_plus_initial_z']))
    summary = dict(protocol=p, seeds=results, trained_z_advantage_seeds=int(sum(descriptive)),
                   trained_z_descriptive_screen=bool(sum(descriptive) >= 4),
                   checks=dict(input_and_checkpoint_hashes_unchanged=True, window_indexing=True,
                               disjoint_probe_date_groups=True, policy_weights_unchanged=True,
                               test_evaluated=False, new_rl_training=False), elapsed_seconds=time.monotonic()-start)
    common.save(out / 'summary.json', summary)
    common.save(out / 'status.json', dict(phase='complete', seeds_completed=5, elapsed_seconds=time.monotonic()-start))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
