"""학습 수온 범위 밖의 NH4 예측 오류를 구분하는 기록 행동 진단."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'e30'))
from darrow_env import Dynamics, STATE, EXO, ZONES


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def metrics(pred, truth, selected):
    assert pred.shape == truth.shape == selected.shape
    p, y = pred[selected], truth[selected]
    if not len(p):
        return {'n_paths_steps': 0}
    assert np.isfinite(p).all() and np.isfinite(y).all()
    flag, bad = p > 4, y > 4
    tp, fp = int((flag & bad).sum()), int((flag & ~bad).sum())
    fn, tn = int((~flag & bad).sum()), int((~flag & ~bad).sum())
    assert tp + fp + fn + tn == len(p)
    return dict(n_paths_steps=len(p), mae=float(abs(p-y).mean()), bias=float((p-y).mean()),
                absolute_error_sum=float(abs(p-y).sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                fpr_pct=100*fp/(fp+tn) if fp+tn else None,
                recall_pct=100*tp/(tp+fn) if tp+fn else None,
                observed_violation_pct=float(100*bad.mean()), predicted_violation_pct=float(100*flag.mean()))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    paths = [Path(protocol[k]) for k in ['prepared', 'model']]
    paths += [HERE/'protocol.json', Path(__file__), HERE.parent/'e30/darrow_env.py']
    paths += [Path(protocol['replay'])/f'{s}_signals.npz' for s in ['train', 'val']]
    paths += [Path(protocol['replay'])/'manifest.json', Path(protocol['replay'])/'results.json']
    hashes = {str(p): sha(p) for p in paths}
    assert sklearn.__version__ == '1.7.2'
    old_manifest = json.loads((Path(protocol['replay'])/'manifest.json').read_text())
    for key in ['prepared', 'model']:
        path = protocol[key]
        assert hashes[path] == old_manifest['input_and_source_sha256'][path]
    save(out/'manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
         commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         input_and_source_sha256=hashes, protocol=protocol, sklearn=sklearn.__version__,
         numpy=np.__version__, python=sys.executable, test_rows_extracted=False))
    start = time.monotonic()
    data = pd.read_pickle(protocol['prepared'])
    splits = {s: list(data['days'][s]) for s in ['train', 'val']}
    assert [len(splits[s]) for s in ['train','val']] == [188,40]
    assert not set(splits['train']) & set(splits['val'])
    # test 날짜 메타데이터와만 겹침을 확인하며 test 행은 추출하지 않는다.
    assert not set(splits['train']+splits['val']) & set(data['days']['test'])
    days = {d:data['data'].loc[d] for s in splits for d in splits[s]}
    train_t = np.concatenate([days[d]['temp'].to_numpy()[:-1] for d in splits['train']])
    lo, hi = float(train_t.min()), float(train_t.max())
    q25, q75 = np.quantile(train_t,[.25,.75])
    dyn = Dynamics(protocol['model'])
    models = [m[0] for m in dyn.models]
    result = dict(train_input_temp_min=lo, train_input_temp_max=hi,
                  train_temperature_quartiles=[float(q25),float(q75)], splits={},
                  test_evaluated=False, weight_updates=0)
    ti = len(STATE) + EXO.index('temp')
    assert ti == 6 and len(models) == 5
    for split, dates in splits.items():
        with np.load(Path(protocol['replay'])/f'{split}_signals.npz') as cache:
            assert cache['days'].tolist() == dates
            recursive, truth = cache['pred'].copy(), cache['actual'].copy()
        assert recursive.shape == truth.shape == (len(dates),5,95)
        temps, xs, current, ys = [], [], [], []
        for d in dates:
            g=days[d];assert len(g)==96
            state=g[STATE].to_numpy(float);exo=g[EXO].to_numpy(float);zone=g[ZONES].to_numpy(float)
            prev=np.vstack([zone[:1],zone[:-2]])
            xs.append(np.column_stack([state[:-1],exo[:-1],zone[:-1]/1000,prev/1000]))
            temps.append(exo[:-1,1]);current.append(state[:-1,0]);ys.append(state[1:,0])
        temp=np.stack(temps);x=np.concatenate(xs);current=np.concatenate(current)
        expected=np.repeat(np.stack(ys)[:,None,:],5,axis=1)
        np.testing.assert_array_equal(truth,expected)
        delta=np.stack([m.predict(x) for m in models])
        one=np.clip(current[None,:]+delta,0,dyn.hi[0]).reshape(5,len(dates),95).transpose(1,0,2)
        groups={'all':np.ones_like(temp,dtype=bool),'below_train':temp<lo,
                'inside_train':(temp>=lo)&(temp<=hi),'above_train':temp>hi}
        groups['never_below_day'] = np.repeat((temp.min(1)>=lo)[:,None],95,axis=1)
        groups['no_low_so_far'] = ~np.maximum.accumulate(temp<lo,axis=1)
        assert sum(int(groups[k].sum()) for k in ['below_train','inside_train','above_train'])==temp.size
        summary=dict(n_days=len(dates), n_transitions=temp.size, input_temp_min=float(temp.min()),
                     input_temp_max=float(temp.max()), all_96_row_temp_min=float(min(days[d].temp.min() for d in dates)),
                     all_96_row_temp_max=float(max(days[d].temp.max() for d in dates)),
                     group_steps={k:int(v.sum()) for k,v in groups.items()}, predictions={})
        for mode,pred in [('observed_one_step',one),('recursive',recursive)]:
            rows={k:metrics(pred,truth,np.repeat(mask[:,None,:],5,axis=1)) for k,mask in groups.items()}
            for name in ['below_train','inside_train','above_train']:
                row=rows[name]
                if row['n_paths_steps']:
                    for key in ['absolute_error_sum','fp','fn']:
                        row[key+'_share_pct']=100*row[key]/rows['all'][key] if rows['all'][key] else None
            for key in ['n_paths_steps','tp','fp','fn','tn','absolute_error_sum']:
                assert np.isclose(sum(rows[k].get(key,0) for k in ['below_train','inside_train','above_train']),rows['all'][key])
            summary['predictions'][mode]=rows
        # 실제 관측 입력에서 수온만 바꾼 모델 반응이다. 인과 효과의 측정이 아니다.
        below=x[:,ti]<lo
        if below.any():
            clipped=x[below].copy();clipped[:,ti]=lo
            maxdiff=max(float(abs(m.predict(x[below])-m.predict(clipped)).max()) for m in models)
            summary['below_train_vs_floor_max_delta_difference']=maxdiff
        low=x.copy();high=x.copy();low[:,ti]=q25;high[:,ti]=q75
        change=np.stack([m.predict(high)-m.predict(low) for m in models])
        summary['temperature_intervention']=dict(low=float(q25),high=float(q75),
            high_minus_low_nh4_delta_mean=float(change.mean()),
            increasing_nh4_pct=float(100*(change>1e-10).mean()),
            decreasing_nh4_pct=float(100*(change < -1e-10).mean()),
            zero_pct=float(100*(abs(change)<=1e-10).mean()))
        curve=[]
        for target in np.linspace(lo,hi,8):
            changed=x.copy();changed[:,ti]=target
            response=np.stack([m.predict(changed) for m in models])
            curve.append(dict(temperature=float(target),mean_nh4_delta=float(response.mean())))
        summary['temperature_response_curve']=curve
        summary['never_below_day_count']=int((temp.min(1)>=lo).sum())
        result['splits'][split]=summary
        print(split,json.dumps(summary),flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    result['hashes_unchanged']=True
    result['elapsed_seconds']=time.monotonic()-start
    save(out/'results.json',result)
    print('DONE',result['elapsed_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
