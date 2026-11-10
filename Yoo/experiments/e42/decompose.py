"""사후 설명용 혼합 오차의 정확한 대수 분해. 추론 규칙은 바꾸지 않는다."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(out):
    z=np.load(out/'predictions.npz');result={'posthoc_only':True,'no_new_predictions_or_thresholds':True,'splits':{}}
    for mode in ('cold','holdout','warm','one_step'):
        a=z['one_step_actual'] if mode=='one_step' else z['actual_'+mode]
        p=z['split_'+mode][...,0]
        truth=np.broadcast_to(a[:,None,:,0],p.shape)
        parts=z['split_parts_'+mode];g,low,high=parts[...,0],parts[...,1],parts[...,2]
        rows={}
        for group,mask in (('normal',truth<=4),('violation',truth>4)):
            if group=='normal':
                expert=low;mixture_term=g*(high-low)
            else:
                expert=high;mixture_term=-(1-g)*(high-low)
            assert np.allclose((p-truth)[mask],((expert-truth)+mixture_term)[mask],atol=1e-12,rtol=0)
            rows[group]={'observed_mean':float(truth[mask].mean()),'predicted_mean':float(p[mask].mean()),
                         'mixture_bias':float((p-truth)[mask].mean()),
                         'corresponding_expert_bias':float((expert-truth)[mask].mean()),
                         'mixture_contribution_to_bias':float(mixture_term[mask].mean()),
                         'corresponding_expert_mae':float(np.abs(expert-truth)[mask].mean()),
                         'mean_g':float(g[mask].mean())}
        result['splits'][mode]=rows
    result['interpretation']='Truth selects an evaluation subgroup and algebraic term only, never a deployed expert. Terms sum to realized bias, but this is not a causal allocation of model error or a proof that a perfect gate is attainable.'
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'decomposition.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['splits']['cold'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
