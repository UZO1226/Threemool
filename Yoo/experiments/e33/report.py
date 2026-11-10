"""두 제약을 따로 집계하고 e30의 같은 seed도 동일 기준으로 다시 비교한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dual import METHODS,LIMITS,save
from common import sha


def main(root,prepared):
    manifest=json.loads((root/'manifest.json').read_text())
    assert all(sha(prepared/k)==v for k,v in manifest['input_sha256'].items())
    prior=torch.load(prepared/'prior.pt',weights_only=True)
    norm=json.loads((prepared/'normalizer.json').read_text())
    rows=[]
    for method in METHODS:
        for seed in manifest['seeds']:
            run=root/'runs'/method/f'seed_{seed}'
            d=json.loads((run/'metrics.json').read_text())
            assert d['steps']==manifest['steps_per_run'] and d['normalizer_and_prior_frozen']
            assert json.loads((run/'model_final/normalizer.json').read_text())==norm
            weights=torch.load(run/'model_final/prior.pt',weights_only=True)
            assert all(torch.equal(v,weights[k]) for k,v in prior.items())
            rows.append({'condition':'separate','method':method,'seed':seed,**d['val_final']})
            old=json.loads((prepared.parent/'main/runs'/method/f'seed_{seed}'/'metrics.json').read_text())
            v=old['val_final'];n,dv=v['nh4_viol_pct']*.95,v['do_viol_pct']*.95
            rows.append({'condition':'sum_e30','method':method,'seed':seed,'air_m3':v['air_m3'],
                         'nh4_cost':n,'do_cost':dv,'cost':n+dv,'both_mean_limits':bool(n<=6 and dv<=6),
                         'steps':old['steps']})
    summary=[]
    md=['# e33 NH4/DO 분리 제약 시험','',f"신규 각{manifest['steps_per_run']:,}전이, seed{manifest['seeds']}. 검증40일 x 구성원5개, 마지막 정책.",
        '기존 e30 합산 조건은 같은 seed를 참고로 재집계했다. 시험 실행(smoke)의 학습량은 기존 조건과 다르므로 우열을 비교하지 않는다.',
        '','| 조건 | 방법 | NH4 비용 | DO 비용 | 공기량 Nm³/23.75h | 두 평균 한도 충족 seed |',
        '| --- | --- | --- | --- | --- | --- |']
    for condition in ('sum_e30','separate'):
        for method in METHODS:
            subset=[r for r in rows if r['condition']==condition and r['method']==method]
            item={'condition':condition,'method':method,'n_seeds':len(subset),
                  'both_limits_seeds':sum(r['both_mean_limits'] for r in subset)}
            for k in ('nh4_cost','do_cost','air_m3'):
                values=[r[k] for r in subset]
                item[k]={'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)) if len(values)>1 else 0.}
            summary.append(item)
            values=[f"{item[k]['mean']:.2f} ± {item[k]['sd']:.2f}" for k in ('nh4_cost','do_cost','air_m3')]
            md.append(f"| {condition} | {method} | {' | '.join(values)} | {item['both_limits_seeds']}/{len(subset)} |")
    md += ['','한도는 학습 실측 평균을 각각 올림한 NH4=6, DO=6이다. 각각의 평균 제약이며 매 에피소드 안전 보장은 아니다.',
           '두 한도 중 하나라도 넘으면 합계가12보다 작아도 미충족으로 판단한다.',
           'seed2개 탐색 결과로 최종 채택이나 통계적 우위를 주장하지 않는다. 환경 편향과 실제 공정 검증 한계는 유지한다.','']
    save(root/'summary.json',{'summary':summary,'runs':rows,'automatic_adoption':False})
    (root/'results.md').write_text('\n'.join(md))
    print('\n'.join(md),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    a=p.parse_args();main(a.root,a.prepared)
