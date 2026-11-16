"""e63 사전 검사.

T1 준비물: 데이터, prior, 정규화는 e50 과 같고 대리 모델은 e51 수정 모델. 기록 재생이 e62 의 DO 수정 모델 값과 정확히 같다.
T2 배분 함수: a1 <= 0 이면 기록 배분 그대로, a1 = 1 이면 w = 0.5, 행 합 1.
T3 새 평가 함수가 1차원 정책에서 e59 평가 함수와 같은 값을 낸다 (차단 없음 전체, park_e19 3일).
T4 a1 = -1 로 고정한 2차원 행동은 같은 총 공기의 1차원 행동과 같은 궤적을 낸다 (park_e10, 3일).
T5 차단이 정책 배분을 본다: 공기 0 정책 + park_e10 에서 a1 = 1 과 a1 = -1 의 실행 공기가 다르다.
T6 2차원 초기 정책: 총 공기 축은 같은 seed 1차원 정책과 정확히 같고 배분 축은 0.
T7 진입점: 배분 조건 2개, 총 공기 조건 2개를 실제 명령줄로 1갱신 학습과 평가, 교차 평가 2개. 결과는 본 실험에 쓰지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alloc as al  # noqa: E402
t60, se, base, h = al.t60, al.se, al.base, al.h
E62 = Path('/home/jio/fcsrl_src/results/e62/main/result.json')
E60_PPO0 = Path('/home/jio/fcsrl_src/results/e60/main/runs/ppo/seed_0/model_final')


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x.item() if isinstance(x, np.generic) else x


def hashes(prepared):
    files = [prepared / 'data.pkl', prepared / 'surrogate.pkl', prepared / 'prior.pt', prepared / 'normalizer.json', se.PREPARED / 'surrogate.pkl',
             se.HERE / 'min_brake.py', se.HERE / 'shield_eval.py', HERE.parent / 'e60/train_shielded.py', HERE / 'alloc.py', HERE / 'train_alloc.py',
             HERE / 'eval_cross.py', HERE.parent / 'e56/hybrid.py', HERE.parent / 'e56/train.py', HERE.parent / 'e30/common.py',
             HERE.parent / 'e30/darrow_env.py', HERE.parent / 'e23/behavior.py']
    return {str(p): base.sha(p) for p in files}


def same(r1, r2, keys=('air_m3', 'cost', 'nh4_cost', 'do_cost')):
    return all(abs(r1[k] - r2[k]) <= 1e-9 * max(1., abs(r2[k])) for k in keys)


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    P = a.prepared
    C = {}
    # T1
    man = json.loads((P / 'e63_manifest.json').read_text())
    splits, days = base.load_selected(P / 'data.pkl')
    dyn = base.Dynamics(P / 'surrogate.pkl')
    env = base.BatchEnv(days, dyn)
    e62 = json.loads(E62.read_text())['B_rollouts']
    rec = {}
    for split in ('val', 'train'):
        r, *_ = al.rollout(env, splits[split], lambda o: env.historical_action(), None)
        rec[split] = r
    C['T1'] = dict(manifest=man, val_cost=rec['val']['cost'], train_cost=rec['train']['cost'],
                   passed=bool(abs(rec['val']['cost'] - e62['do_corrected|val|record|record']['cost']) < 1e-9
                               and abs(rec['train']['cost'] - e62['do_corrected|train|record|record']['cost']) < 1e-9
                               and man['surrogate.pkl']['sha256'] == base.sha(P / 'surrogate.pkl')))
    # T2
    env.reset(al.all_episodes(splits['val'][:2], 5))
    recf = env.x['frac'].copy()
    ok = True
    for a1, w in ((-1., 0.), (-.3, 0.), (0., 0.), (.5, .25), (1., .5), (1.7, .5)):
        got = al.apply_allocation(env, recf, np.full(len(recf), a1))
        frac = env.x['frac'][:, env.t]
        ok &= bool(np.allclose(got, w) and np.allclose(frac, (1 - w) * recf[:, env.t] + w * al.BACK) and np.allclose(frac.sum(1), 1))
        env.x['frac'][:, env.t] = recf[:, env.t]
    C['T2'] = dict(passed=ok)
    # T3
    common = base.Dynamics(se.PREPARED / 'surrogate.pkl')
    cenv = base.BatchEnv(days, common)
    c = base.common.config(66)
    p1 = base.common.make_agent('ppo', 0, se.PREPARED, c)
    p1.load(E60_PPO0)
    act1 = lambda o: p1.act_det(o)[:, 0]  # noqa: E731
    r_new, *_ = al.rollout(cenv, splits['val'], act1, None)
    r_old, *_ = se.rollout(cenv, splits['val'], act1, None)
    r_new19, *_ = al.rollout(cenv, splits['val'][:3], act1, se.make_shield('park_e19', common))
    r_old19, *_ = se.rollout(cenv, splits['val'][:3], act1, se.make_shield('park_e19', common))
    C['T3'] = dict(none=[r_new['cost'], r_old['cost']], park_e19_3days=[r_new19['cost'], r_old19['cost']],
                   passed=bool(same(r_new, r_old) and same(r_new19, r_old19)))
    # T4
    r1, _, s1, a1_, z1 = al.rollout(env, splits['val'][:3], act1, se.make_shield('park_e10', dyn))
    r2, _, s2, a2_, z2 = al.rollout(env, splits['val'][:3], lambda o: np.column_stack([act1(o), np.full(len(o), -1.)]), se.make_shield('park_e10', dyn))
    C['T4'] = dict(passed=bool(np.array_equal(s1, s2) and np.array_equal(a1_, a2_) and np.array_equal(z1, z2)))
    # T5
    zero = lambda w: (lambda o: np.column_stack([np.full(len(o), -1.), np.full(len(o), w)]))  # noqa: E731
    rz0, _, _, az0, zz0 = al.rollout(env, splits['val'][:3], zero(-1.), se.make_shield('park_e10', dyn))
    rz1, _, _, az1, zz1 = al.rollout(env, splits['val'][:3], zero(1.), se.make_shield('park_e10', dyn))
    C['T5'] = dict(record_alloc=dict(air=rz0['air_m3'], nh4=rz0['nh4_cost'], do=rz0['do_cost']),
                   back_half=dict(air=rz1['air_m3'], nh4=rz1['nh4_cost'], do=rz1['do_cost'], w_mean=rz1['w_mean']),
                   share_steps_air_differs=float((np.abs(az0 - az1) > 1e-6).mean()), passed=bool((np.abs(az0 - az1) > 1e-6).any() and rz1['w_mean'] == .5))
    # T6
    obs = np.concatenate([cenv.reset(al.all_episodes(splits['train'][:20], 5))])
    t6 = {}
    for m in ('ppo', 'fcsrl'):
        one = base.common.make_agent(m, 3, P, c)
        two = al.make_agent(m, 3, P, c)
        x1, x2 = one.act_det(obs), two.act_det(obs)
        t6[m] = bool(np.array_equal(x1[:, 0], x2[:, 0]) and np.all(x2[:, 1] == 0))
    C['T6'] = dict(**t6, passed=all(t6.values()))
    # T7
    smoke = a.out / 'smoke'
    base.save(a.out / 'smoke_manifest.json', dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(), hashes=hashes(P)))
    env_vars = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
                    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    jobs = {}
    for cond in ('alloc', 'total'):
        for m in ('ppo', 'fcsrl'):
            script = HERE / 'train_alloc.py' if cond == 'alloc' else HERE.parent / 'e60/train_shielded.py'
            cmd = [sys.executable, str(script), '--method', m, '--seed', '0', '--updates', '1', '--prepared', str(P),
                   '--out', str(smoke / cond / m), '--manifest', str(a.out / 'smoke_manifest.json')]
            jobs[(cond, m)] = (time.monotonic(), subprocess.Popen(cmd, env=env_vars, stdout=(a.out / f'smoke_{cond}_{m}.log').open('w'), stderr=subprocess.STDOUT))
    for (cond, m), (t0, proc) in jobs.items():
        code = proc.wait()
        entry = dict(exit=code, seconds=time.monotonic() - t0)
        if code == 0:
            mt = json.loads((smoke / cond / m / 'metrics.json').read_text())
            entry.update(evaluations=sorted(k for k in mt if isinstance(mt[k], dict) and k != 'val_final'),
                         val_park_e19_cost=mt['val_park_e19']['cost'], val_park_e19_w_mean=mt['val_park_e19'].get('w_mean'))
        C[f'T7_{cond}_{m}'] = entry
    cross = {}
    for cond in ('alloc', 'total'):
        if C[f'T7_{cond}_ppo']['exit'] == 0:
            cmd = [sys.executable, str(HERE / 'eval_cross.py'), '--condition', cond, '--method', 'ppo', '--seed', '0', '--run-dir', str(smoke / cond / 'ppo'),
                   '--prepared', str(P), '--out', str(smoke / 'cross' / cond)]
            cross[cond] = subprocess.Popen(cmd, env=env_vars, stdout=(a.out / f'smoke_cross_{cond}.log').open('w'), stderr=subprocess.STDOUT)
    for cond, proc in cross.items():
        C[f'T7_cross_{cond}'] = dict(exit=proc.wait())
    shutil.rmtree(smoke, ignore_errors=True)
    t7 = all(C[f'T7_{c}_{m}']['exit'] == 0 and len(C[f'T7_{c}_{m}']['evaluations']) == 5 for c in ('alloc', 'total') for m in ('ppo', 'fcsrl'))
    t7 = t7 and all(C.get(f'T7_cross_{c}', {}).get('exit') == 0 for c in ('alloc', 'total'))
    C['passed'] = bool(all(C[k]['passed'] for k in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')) and t7)
    C['hashes'] = hashes(P)
    C = plain(C)
    base.save(a.out / 'checks.json', C)
    print(json.dumps({k: v for k, v in C.items() if k != 'hashes'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
