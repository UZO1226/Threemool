"""모델 분리, 출력 결합과 저장된 지표를 독립 검산한다."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score,brier_score_loss
from regime import NextNH4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out,source):
    manifest=json.loads((out/'manifest.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    c=json.loads((out/'comparison.json').read_text())
    source_manifest=json.loads((source/'manifest.json').read_text())
    source_protocol=json.loads((source/'protocol.json').read_text())
    assert sha(source/'protocol.json')==source_manifest['protocol_sha256']
    for key in ('fit_dates','holdout_dates','validation_dates'):
        assert protocol[key]==source_protocol[key]
    fit,holdout,val=[protocol[k] for k in ('fit_dates','holdout_dates','validation_dates')]
    assert (len(fit),len(holdout),len(val))==(140,48,40)
    assert max(fit)<min(holdout) and max(holdout)<min(val)
    assert not(set(fit)&set(holdout) or set(holdout)&set(val) or set(fit)&set(val))
    with (out/'candidate.pkl').open('rb') as f:new=pickle.load(f)
    with (source/'candidate.pkl').open('rb') as f:old=pickle.load(f)
    assert pickle.dumps(new['base'])==pickle.dumps(old['base'])
    for name in ('control','augmented'):
        assert pickle.dumps(new['nh4_models'][name])==pickle.dumps(old['nh4_models'][name])
    examples=np.load(source/'training_examples.npz');ids=np.load(source/'training_row_ids.npy')
    for m in range(5):
        target=examples['observed_x'][ids[m],0]+examples['observed_y'][ids[m]]
        positive=int((target>4).sum())
        tree_budget=old['nh4_models']['augmented'][m].n_iter_
        split=new['nh4_models']['split'][m]
        assert isinstance(split,NextNH4)
        assert split.gate.class_weight is None and split.gate.loss=='log_loss'
        assert np.array_equal(split.gate.classes_,[False,True])
        for model in (split.gate,split.normal,split.violation):
            assert model.n_iter_==tree_budget and model.early_stopping is False
            assert np.array_equal(model.monotonic_cst,old['nh4_models']['augmented'][m].monotonic_cst)
        assert new['nh4_models']['direct'][m].single.n_iter_==tree_budget
        assert new['nh4_models']['capacity'][m].single.n_iter_==3*tree_budget
        rows=next(v for v in c['training'] if v['arm']=='split' and v['member']==m)
        assert rows['positive_rows']==2*positive and rows['positive_unique_rows']==positive
        assert rows['positive_dates']==len(np.unique((ids[m]//95)[target>4]))
        assert rows['trees']==3*tree_budget
    z=np.load(out/'predictions.npz');prev=np.load(source/'predictions.npz')
    checks=0
    for name in c['results']:
        for mode in ('holdout','cold','warm','one_step'):
            p=z[name+'_'+mode]
            actual=z['one_step_actual'] if mode=='one_step' else z['actual_'+mode]
            truth=np.broadcast_to(actual[:,None],p.shape)
            assert np.isfinite(p).all() and (p>=0).all() and (p<=new['base']['hi']).all()
            v=c['results'][name][mode]
            if name in ('control','augmented'):assert np.array_equal(p,prev[name+'_'+mode])
            for j,state in enumerate(('nh4','no3','do1','do2','do3')):
                assert np.isclose(np.abs(p[...,j]-truth[...,j]).mean(),v['mae'][state])
                assert np.isclose(np.sqrt(((p[...,j]-truth[...,j])**2).mean()),v['rmse'][state])
                checks+=2
            for target,t,a in (('NH4',truth[...,0]>4,p[...,0]>4),('DO',truth[...,2:].max(3)>2,p[...,2:].max(3)>2)):
                tn,fp,fn,tp=confusion_matrix(t.ravel(),a.ravel(),labels=[False,True]).ravel()
                q=v['confusion'][target]
                assert [int(tp),int(fp),int(tp+fn),int(tn+fp)]==[q[k] for k in ('tp','fp','positives','negatives')]
                assert np.isclose(q['recall_pct'],100*tp/(tp+fn))
                assert np.isclose(q['fpr_pct'],100*fp/(fp+tn))
                assert np.isclose(np.abs(t.sum(2)-a.sum(2)).mean(),v['counts'][target]['mae'])
                checks+=3
            for group,mask in (('normal',truth[...,0]<=4),('violation',truth[...,0]>4)):
                q=c['group_errors'][name][mode][group]
                assert q['samples']==mask.sum()
                assert np.isclose(np.abs(p[...,0][mask]-truth[...,0][mask]).mean(),q['mae'])
                assert np.isclose((p[...,0][mask]-truth[...,0][mask]).mean(),q['bias'])
                checks+=2
            if name=='split':
                parts=z['split_parts_'+mode]
                g,low,high=parts[...,0],parts[...,1],parts[...,2]
                assert ((g>=0)&(g<=1)).all()
                assert ((low>=0)&(low<=4)).all()
                assert ((high>=4)&(high<=new['base']['hi'][0])).all()
                assert (low<=high).all()
                assert np.allclose(p[...,0],low+g*(high-low),atol=1e-12,rtol=0)
                t=truth[...,0]>4
                q=c['gate_diagnostics'][mode]
                assert np.isclose(roc_auc_score(t.ravel(),g.ravel()),q['auc'])
                assert np.isclose(brier_score_loss(t.ravel(),g.ravel()),q['brier'])
                checks+=3
        assert np.array_equal(z[name+'_one_step'][...,1:],z['control_one_step'][...,1:])
        for e in c['events'][name]:
            step=e['minutes']//15;a=z['event_actual']
            mask=z['event_dq']>0 if e['group']=='increase' else z['event_dq']<0
            if e['group']=='decrease_rising':mask &= a[:,step,0]-a[:,0,0]>.1
            p=z[name+'_events'][mask,:,step,0]
            t=np.broadcast_to(a[mask,step,0][:,None],p.shape)
            assert e['windows']==int(mask.sum())
            assert np.isclose(np.abs(p-t).mean(),e['nh4_mae'])
            tn,fp,fn,tp=confusion_matrix((t>4).ravel(),(p>4).ravel(),labels=[False,True]).ravel()
            assert [int(tp),int(fp),int(tp+fn),int(tn+fp)]==[e['confusion'][k] for k in ('tp','fp','positives','negatives')]
            checks+=2
    assert all(sha(path)==value for path,value in manifest['input_sha256'].items())
    assert sha(out/'protocol.json')==manifest['protocol_sha256']
    assert sha(out/'candidate.pkl')==manifest['candidate_sha256_before_eval']
    for name,digest in manifest['source_sha256'].items():
        assert sha(Path(__file__).parent/name)==digest
    checked={'passed':True,'independent_metric_checks':checks,'ordered_experts_and_mixture_identity':True,
             'fixed_models_and_inputs':True,'tree_budgets_and_class_support':True,'date_separation':True}
    (out/'verification.json').write_text(json.dumps(checked,indent=2))
    print(checked)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();main(a.out,a.source)
