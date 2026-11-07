"""정책 학습 전에 초기 상태와 훈련 관측 통계를 고정한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.integrate import ODEintWarning
from tqdm import tqdm
from bsm_mean import MeanEnv, snapshot, evaluate

p = argparse.ArgumentParser()
p.add_argument('--steady', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
warnings.filterwarnings('error', category=ODEintWarning)
started = time.monotonic()
state = dict(np.load(a.steady))
env = MeanEnv(state, days=14.)
env.reset()
for _ in tqdm(range(env.T), desc='dry conditioning 14 days', mininterval=5):
    env.step(np.array([84 / 120 - 1]))
initial = snapshot(env.model)
np.savez_compressed(a.out / 'initial.npz', **initial)
env = MeanEnv(initial, days=7.)
obs, _ = env.reset()
observations, flow_means = [], []
flows = []
for t in tqdm(range(env.T), desc='training states dry days 0 to 6', mininterval=5):
    if t % 288 == 0:
        np.savez_compressed(a.out / f'day_{t // 288}.npz', **snapshot(env.model))
    observations.append(obs.copy())
    obs, _, _, _, info = env.step(np.array([84 / 120 - 1]))
    flows.append(info['flow_ratio'])
    if (t + 1) % 288 == 0:
        flow_means.append(float(np.mean(flows)))
        flows = []
np.save(a.out / 'train_observations.npy', np.asarray(observations))
metadata = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
    'created_at': datetime.now(timezone.utc).isoformat(),
    'steady_sha256': hashlib.sha256(a.steady.read_bytes()).hexdigest(),
    'archive_sha256': hashlib.sha256(Path(os.environ['BSM1_ARCHIVE']).read_bytes()).hexdigest(),
    'training_days': list(range(7)), 'daily_reference_flow_ratio': flow_means,
    'files': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(a.out.glob('*.np*'))}}
(a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
refs = {}
for name, policy, controller in [('fixed84', lambda o: [84 / 120 - 1], False),
                                  ('fixed240', lambda o: [1.], False),
                                  ('do_pi', lambda o: [0.], True)]:
    refs[name] = evaluate(initial, policy, controller)
    print(name, {w: v['week'] for w, v in refs[name].items()}, flush=True)
    (a.out / 'references.json').write_text(json.dumps(refs, indent=2, allow_nan=False))
print('PREPARED', time.monotonic() - started, flush=True)
