"""e50: 공통 평가 조건 v1.1 기준 대리 모델에서 Yoo 의 세 조건(PPO-Lag, FCSRL + PPO-Lag, FCSRL + Actor 보정)을 다시 학습한다

학습과 평가 코드는 e30 의 train.py 를 수정 없이 부른다. 바뀐 것은 준비물(scikit-learn 1.9.1 로 만든 대리 모델과
같은 절차의 초기 정책)과 조건 수뿐이다. 준비물은 e30 prepare.py 로 만들고, 기록 운전 재생이 공통 검산값과 맞는지
먼저 확인한다.

    python run.py --prepared /home/jio/fcsrl_src/results/e50/prepared --out /home/jio/fcsrl_src/results/e50/main --jobs 6
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
E30 = HERE.parent / 'e30'
sys.path.insert(0, str(E30))
from common import save, sha  # noqa: E402

METHODS = ('ppo', 'fcsrl', 'fcsrl_actor')
CHECK = {'val': {'cost': 28.77, 'nh4_pct': 28.48, 'do_pct': 1.80}, 'train': {'cost': 12.17}}


def check_replay(prepared):
    """PROTOCOL v1.1 의 기준 모델 검산값과 비교한다"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('e31_diagnose', HERE.parent / 'e31' / 'diagnose.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    replay = mod.replay
    from darrow_env import BatchEnv, Dynamics, load_days
    splits, day_data = load_days(prepared / 'data.pkl')
    env = BatchEnv(day_data, Dynamics(prepared / 'surrogate.pkl'))
    out = {}
    for split, ref in CHECK.items():
        r = replay(env, splits[split])
        got = {'cost': float(r.cost.mean()), 'nh4_pct': float(r.nh4_pct.mean()), 'do_pct': float(r.do_pct.mean()),
               'air_m3': float(r.air_m3.mean())}
        out[split] = {'value': got, 'reference': ref,
                      'passed': all(abs(round(got[k], 2) - v) <= .011 for k, v in ref.items())}
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def main(a):
    import sklearn
    check = check_replay(a.prepared)
    check['sklearn'] = sklearn.__version__
    print('REPLAY CHECK', json.dumps(check, ensure_ascii=False), flush=True)
    assert check['passed'], '기준 모델 검산값과 다르다'
    a.out.mkdir(parents=True, exist_ok=False)
    save(a.out / 'replay_check.json', check)
    repo = HERE.parents[2]
    upstream = Path('/home/jio/fcsrl_src/FCSRL')
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
                'python': sys.executable, 'sklearn': sklearn.__version__, 'parallel_jobs': a.jobs,
                'methods': METHODS, 'seeds': list(range(a.seeds)), 'updates': a.updates,
                'steps_per_run': a.updates * 6080, 'evaluation_split': 'val', 'selection': 'last',
                'input_sha256': {p.name: sha(p) for p in a.prepared.iterdir() if p.is_file()}, 'commands': []}
    pending = [(m, s) for s in range(a.seeds) for m in METHODS]
    active, complete = [], []
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=f"{repo / 'Yoo/code/wwtp'}:{E30}")

    def status(phase):
        save(a.out / 'status.json', {'phase': phase, 'completed': complete, 'pending': pending,
             'running': [{'method': m, 'seed': s, 'pid': p.pid} for p, _, m, s in active],
             'updated_at': datetime.now(timezone.utc).isoformat()})

    def stop(signum, frame):
        raise InterruptedError(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            while pending and len(active) < a.jobs:
                m, s = pending.pop(0)
                cmd = [sys.executable, '-u', str(E30 / 'train.py'), '--method', m, '--seed', str(s), '--updates',
                       str(a.updates), '--prepared', str(a.prepared), '--out', str(a.out / 'runs' / m / f'seed_{s}')]
                log = (a.out / f'{m}_s{s}.log').open('w')
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                active.append((proc, log, m, s))
                manifest['commands'].append(cmd)
                save(a.out / 'manifest.json', manifest)
                print(f'[{time.strftime("%H:%M")}] START', m, s, flush=True)
            status('training')
            time.sleep(5)
            for item in active.copy():
                p, log, m, s = item
                code = p.poll()
                if code is None:
                    continue
                log.close()
                active.remove(item)
                complete.append({'method': m, 'seed': s, 'exit_code': code})
                print(f'[{time.strftime("%H:%M")}] END', m, s, code, f'{len(complete)}/{len(METHODS) * a.seeds}', flush=True)
                if code:
                    raise RuntimeError(f'{m} seed {s} failed')
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(a.out / 'manifest.json', manifest)
        status('complete')
    except BaseException:
        for proc, log, _, _ in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            log.close()
        status('failed')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--updates', type=int, default=66)
    main(p.parse_args())
