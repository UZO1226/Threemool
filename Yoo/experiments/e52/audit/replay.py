"""저장된 e50/e52 정책을 원 환경에서 재평가하고 시간별 진단을 보존한다. 학습 없음."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import sklearn
import torch

HERE = Path(__file__).resolve().parent
YOO = HERE.parents[2]
EXP = YOO / 'experiments'
ROOT = YOO.parent
sys.path.insert(0, str(EXP / 'e30'))
import common
sys.path.insert(0, str(EXP / 'e45'))
from evaluate import load_selected, fingerprint, metrics
from command import CommandEnv, make_agent, config
from darrow_env import BatchEnv, Dynamics, all_episodes, action_to_air
from fcsrl.utils import DeviceConfig

RESULTS = Path('/home/jio/fcsrl_src/results')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    common.save(p, x)


def jobs(exp):
    if exp == 'e53':
        return [dict(id=f'{m}_d045_s{s}', method=m, seed=s, delta=0.45,
                     path=str(RESULTS / 'e53/main/runs' / m / 'd0.45' / f'seed_{s}'))
                for m in ('ppo', 'fcsrl') for s in range(3)]
    if exp == 'e50':
        return [dict(id=f'{m}_s{s}', method=m, seed=s, path=str(RESULTS / exp / 'main/runs' / m / f'seed_{s}'))
                for m in ('ppo', 'fcsrl', 'fcsrl_actor') for s in range(5)]
    return [dict(id=f'{m}_k{k}_s{s}', method=m, seed=s, kappa=k,
                 path=str(RESULTS / ('e44/main/runs' if k == 0 and s < 2 else 'e52/main/runs') / m /
                          ('padded' if k == 0 and s < 2 else f'k{k}') / f'seed_{s}'))
            for m in ('ppo', 'fcsrl') for k in (0, 10) for s in range(3)]


def run(a):
    assert sklearn.__version__ == ('1.9.1' if a.experiment == 'e50' else '1.7.2')
    a.out.mkdir(parents=True, exist_ok=False)
    prepared = RESULTS / ('e50' if a.experiment == 'e50' else 'e30') / 'prepared'
    model = prepared / 'surrogate.pkl' if a.experiment == 'e50' else RESULTS / 'e35/zone_fix/corrected_surrogate.pkl'
    js = jobs(a.experiment)
    inputs = [prepared / n for n in ('data.pkl', 'prior.pt', 'normalizer.json')] + [model]
    for j in js:
        inputs += list((Path(j['path']) / 'model_final').iterdir())
        inputs.append(Path(j['path']) / 'metrics.json')
    sources = list((YOO / 'code/wwtp').glob('*.py'))
    for exp in ('e29', 'e30', 'e33', 'e44', 'e45'):
        sources += list((EXP / exp).glob('*.py'))
    sources.append(Path(__file__))
    hashes = {str(p): sha(p) for p in inputs + sources if p.is_file()}
    splits, days = load_selected(prepared / 'data.pkl')
    manifest = dict(experiment=a.experiment, jobs=js, prepared=str(prepared), dynamics=str(model),
                    dates=splits['val'], hashes=hashes, python=sys.executable, sklearn=sklearn.__version__,
                    numpy=np.__version__, torch=torch.__version__, created_at=datetime.now(timezone.utc).isoformat(),
                    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    selection='all final checkpoints, deterministic, validation only',
                    metric_tolerance=dict(rtol=1e-9,atol=1e-7,air_atol=a.air_atol),
                    test_evaluated=False, weight_updates=0, air_unit='m3/23.75h',
                    disagreement='population SD of five raw delta predictions, before state clipping, identical to e52')
    save(a.out / 'manifest.json', manifest)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    dyn = Dynamics(model)
    start = time.monotonic()
    for index, j in enumerate(js):
        tick = time.monotonic()
        if a.experiment == 'e50':
            agent = common.make_agent(j['method'],j['seed'],prepared,common.config(66))
            env = BatchEnv(days,dyn)
        else:
            agent = make_agent(j['method'],j['seed'],prepared,config(66))
            env = CommandEnv(days,dyn,'padded')
        agent.load(Path(j['path']) / 'model_final')
        before = fingerprint(agent)
        episodes = all_episodes(splits['val'],5)
        obs = env.reset(episodes)
        arrays = {k:[] for k in ('states','pre_states','air','sd')}
        for t in range(95):
            action = agent.act_det(obs)[:,0]
            air = action_to_air(action)
            arrays['pre_states'].append(env.state.copy())
            if a.experiment in ('e52', 'e53'):
                zone = air[:,None] * env.x['frac'][:,t]
                x = np.column_stack([env.state,env.x['exo'][:,t],zone/1000.,env.prev_zone/1000.])
                sd = np.stack([member[0].predict(x) for member in dyn.models]).std(0)
                arrays['sd'].append(sd)
            obs,reward,cost,done,info = env.step(action)
            assert np.allclose(reward,-info['air']/9000.)
            assert np.array_equal(cost,info['v_nh4']+info['v_do'])
            arrays['states'].append(env.state.copy())
            arrays['air'].append(info['air'].copy())
        assert done and before == fingerprint(agent)
        arrays = {k:np.stack(v,1) for k,v in arrays.items() if v}
        summary,rows,members = metrics(arrays['states'],arrays['air'],episodes)
        n = (arrays['states'][:,:,0]>4).sum(1)
        d = (arrays['states'][:,:,2:].max(2)>2).sum(1)
        summary.update(cost=float((n+d).mean()),nh4_viol_pct=float((n*100/95).mean()),
                       do_viol_pct=float((d*100/95).mean()),days_within_limit_pct=float(100*((n+d)<=12).mean()),n_episodes=200)
        previous = json.loads((Path(j['path'])/'metrics.json').read_text())['val_final']
        checked = []
        for key,value in previous.items():
            if key in summary:
                assert np.isclose(summary[key],value,rtol=1e-9,atol=a.air_atol if key=='air_m3' else 1e-7), (j['id'],key,summary[key],value)
                checked.append(key)
        dest = a.out / j['id']
        dest.mkdir()
        np.savez_compressed(dest/'rollout.npz',**arrays)
        save(dest/'episodes.json',rows)
        save(dest/'metrics.json',dict(job=j,summary=summary,members=members,original_metrics_reproduced=checked,
                                     frozen_policy=True,fingerprint=before,seconds=time.monotonic()-tick))
        save(a.out/'status.json',dict(phase='running',completed=index+1,total=len(js),last=j['id'],elapsed=time.monotonic()-start))
        print('DONE',a.experiment,j['id'],round(time.monotonic()-tick,2),flush=True)
    assert all(sha(p)==v for p,v in hashes.items())
    save(a.out/'status.json',dict(phase='complete',completed=len(js),total=len(js),elapsed=time.monotonic()-start,
                                hashes_unchanged=True,test_evaluated=False,weight_updates=0))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',choices=['e50','e52','e53'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--air-atol',type=float,default=1e-7)
    args=p.parse_args()
    try:
        run(args)
    except BaseException as exc:
        if args.out.exists():
            save(args.out/'failure.json',dict(error=repr(exc),phase='failed'))
        raise
