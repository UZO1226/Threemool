"""학습과 평가에 쓸 BSM1 시작 상태를 만든다 (Park 작성).

순서는 팀과 같다 (Yoo/e16, Lee/e01).
1. 일정한 유입으로 200일을 돌려 정상상태를 만들고 MATLAB 기준값과 비교한다
2. 건기 14일을 기본 폭기(KLa5 = 84) 로 돌린 상태를 평가 시작 상태 initial.npz 로 쓴다
3. 그 상태에서 건기 0 ~ 6일을 다시 기본 폭기로 돌리며 하루 시작 상태 day_0 ~ day_6.npz 와
   관측 정규화용 관측값 train_observations.npy 를 저장한다

    export BSM1_ARCHIVE=<BSM1.zip 경로>
    python prepare.py --out <결과 폴더>/prepared
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
from tqdm import tqdm

from env import AerationEnv, FIXED_KLAS, STEPS_PER_DAY, kla_to_action, snapshot

HERE = Path(__file__).resolve().parent

# bsm2-python 의 tests/bsm1_ol_test.py 에 있는 일정 유입과 MATLAB 정상상태 방류값
CONSTANT_INFLUENT = np.array([0., 30, 69.5, 51.2, 202.32, 28.17, 0, 0, 0, 0, 31.56, 6.95,
                              10.59, 7, 211.2675, 18446, 15, 0, 0, 0, 0, 0])
MATLAB_EFFLUENT = np.array([30., .889492799653682, 4.39182747787874, .188440413683379,
                            9.78152406404732, .572507856962265, 1.72830016782928, .490943515687561,
                            10.4152201204309, 1.73333146817512, .688280004678034, .0134804685779854,
                            4.12557938198182, 12.4969499853007, 18061., 15., 0, 0, 0, 0, 0])


def steady_state():
    from bsm2_python import BSM1OL
    last = CONSTANT_INFLUENT.copy()
    last[0] = 200.1
    model = BSM1OL(data_in=np.stack([CONSTANT_INFLUENT, last]), timestep=15 / 1440, endtime=200.)
    for i in tqdm(range(len(model.simtime)), desc='정상상태 200일', mininterval=5):
        model.step(i, FIXED_KLAS.copy())
    error = float(np.abs(model.ys_eff - MATLAB_EFFLUENT).max())
    np.testing.assert_allclose(model.ys_eff, MATLAB_EFFLUENT, rtol=1e-5, atol=1e-5)
    return snapshot(model), error


def run_fixed(env, kla=84.0, on_step=None):
    obs, _ = env.reset()
    action = np.array([kla_to_action(kla)])
    while env.t < env.T:
        if on_step is not None:
            on_step(env, obs)
        obs, *_ = env.step(action)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    try:
        from scipy.integrate import ODEintWarning
        warnings.filterwarnings('error', category=ODEintWarning)
    except ImportError:
        pass

    state, matlab_error = steady_state()
    np.savez_compressed(a.out / 'steady_state.npz', **state)
    print('MATLAB 정상상태 최대 절대 오차', matlab_error, flush=True)

    env = AerationEnv(state, 'dry', start_day=0, days=14)
    run_fixed(env)
    initial = snapshot(env.model)
    np.savez_compressed(a.out / 'initial.npz', **initial)

    observations = []

    def record(env, obs):
        if env.t % STEPS_PER_DAY == 0:
            np.savez_compressed(a.out / f'day_{env.t // STEPS_PER_DAY}.npz', **snapshot(env.model))
        observations.append(obs.copy())

    run_fixed(AerationEnv(initial, 'dry', start_day=0, days=7), on_step=record)
    np.save(a.out / 'train_observations.npy', np.asarray(observations))

    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    except Exception:
        commit = None
    meta = {'code_commit': commit, 'matlab_max_abs_error': matlab_error,
            'archive_sha256': sha256(os.environ['BSM1_ARCHIVE']),
            'source_sha256': {f.name: sha256(f) for f in sorted(HERE.glob('*.py'))},
            'files': {f.name: sha256(f) for f in sorted(a.out.glob('*.np*'))}}
    (a.out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print('완료', a.out, flush=True)


if __name__ == '__main__':
    main()
