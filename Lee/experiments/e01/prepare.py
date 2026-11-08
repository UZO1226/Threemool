"""학습 전에 BSM1 초기 상태, 훈련 관측 통계, 기준 제어 결과를 고정한다.

Yoo/experiments/e16/preflight.py 의 정상상태 절차와 Yoo/experiments/e17/prepare.py 의
조건화 및 관측 통계 절차를 합쳐 Lee 환경에서 다시 수행한다. 두 비용 모드가 같은
초기 상태와 같은 관측 정규화 통계를 쓰도록 한 번만 만든다.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy.integrate import ODEintWarning
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'Lee/code/bsm'))
from bsm2_python import BSM1OL  # noqa: E402
from bsm_binary import BinaryEnv, evaluate  # noqa: E402
from bsm_env import snapshot, BASE_KLAS  # noqa: E402

# upstream tests/bsm1_ol_test.py 의 상수 유입과 MATLAB 기준 유출값 (Yoo/e16 과 동일)
CONSTANT_INFLUENT = np.array([0., 30, 69.5, 51.2, 202.32, 28.17, 0, 0, 0, 0, 31.56, 6.95,
                              10.59, 7, 211.2675, 18446, 15, 0, 0, 0, 0, 0])
MATLAB_EFFLUENT = np.array([30., .889492799653682, 4.39182747787874, .188440413683379,
    9.78152406404732, .572507856962265, 1.72830016782928, .490943515687561,
    10.4152201204309, 1.73333146817512, .688280004678034, .0134804685779854,
    4.12557938198182, 12.4969499853007, 18061., 15., 0, 0, 0, 0, 0])


def steady_state():
    last = CONSTANT_INFLUENT.copy()
    last[0] = 200.1
    model = BSM1OL(data_in=np.stack([CONSTANT_INFLUENT, last]), timestep=15 / 1440, endtime=200.)
    for i in tqdm(range(len(model.simtime)), desc='200 day stabilization', mininterval=5):
        model.step(i, BASE_KLAS.copy())
    np.testing.assert_allclose(model.ys_eff, MATLAB_EFFLUENT, rtol=1e-5, atol=1e-5)
    return snapshot(model), float(abs(model.ys_eff - MATLAB_EFFLUENT).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings('error', category=ODEintWarning)
    started = time.monotonic()

    state, matlab_error = steady_state()
    np.savez_compressed(a.out / 'steady_state.npz', **state)

    # 건기 14일 기본 폭기로 조건화한 상태를 모든 실행의 공통 초기 상태로 쓴다.
    env = BinaryEnv(state, days=14.)
    env.reset()
    for _ in tqdm(range(env.T), desc='dry conditioning 14 days', mininterval=5):
        env.step(np.array([84 / 120 - 1]))
    initial = snapshot(env.model)
    np.savez_compressed(a.out / 'initial.npz', **initial)

    # 학습용 하루 시작 상태 7개와 관측 정규화 통계
    env = BinaryEnv(initial, days=7.)
    obs, _ = env.reset()
    observations, flow_means, flows = [], [], []
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

    metadata = {
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'matlab_reference_max_abs_error': matlab_error,
        'archive_sha256': hashlib.sha256(Path(os.environ['BSM1_ARCHIVE']).read_bytes()).hexdigest(),
        'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / 'Lee/code/bsm').glob('*.py'))},
        'training_days': list(range(7)), 'daily_reference_flow_ratio': flow_means,
        'files': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(a.out.glob('*.np*'))}}
    (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    # 기준 제어. 비용 한도를 정한 근거이며 RL 결과를 보기 전에 측정한다.
    references = {}
    for name, policy, controller in [('fixed84', lambda o: [84 / 120 - 1], False),
                                     ('fixed240', lambda o: [1.], False),
                                     ('do_pi', lambda o: [0.], True)]:
        references[name] = evaluate(initial, policy, controller)
        summary = {w: {k: round(v[w]['week'][k], 3) for k in
                       ('aeration_kwh_per_day', 'flow_weighted_nh4', 'step_violation_percent',
                        'violating_steps_per_day')} for w in ('dry', 'rain') for v in [references[name]]}
        print(name, json.dumps(summary), flush=True)
        (a.out / 'references.json').write_text(json.dumps(references, indent=2, allow_nan=False))
    print('PREPARED', round(time.monotonic() - started, 1), flush=True)


if __name__ == '__main__':
    main()
