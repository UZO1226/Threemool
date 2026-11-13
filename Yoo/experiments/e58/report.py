"""같은 backend의 세 조건을 seed별 대응 비교한다."""
import argparse
import json
from pathlib import Path
import numpy as np

KEYS=['air_m3','nh4_cost','do_cost','cost','nh4_excess_mg_h_L','do_excess_mg_h_L','worst_day_nh4_cost','worst_day_do_cost']


def report(root):
    rows={}
    for method in ['baseline','severity','detached']:
        rows[method]=[json.loads((root/'runs'/method/f'seed_{seed}'/'metrics.json').read_text()) for seed in range(5)]
        assert all(x['steps']==401280 and x['prior_mode']=='on' and x['backend']=='gpu1_update' for x in rows[method])
    means={m:{k:float(np.mean([x['val_final'][k] for x in rs])) for k in KEYS} for m,rs in rows.items()}
    sd={m:{k:float(np.std([x['val_final'][k] for x in rs],ddof=1)) for k in KEYS} for m,rs in rows.items()}
    passed={};differences={}
    rng=np.random.default_rng(580058)
    indices=rng.integers(0,5,size=(10000,5))
    for ref in ['baseline','detached']:
        candidate=rows['severity'];control=rows[ref]
        d={k:np.array([a['val_final'][k]-b['val_final'][k] for a,b in zip(candidate,control)]) for k in KEYS}
        differences[ref]={k:dict(paired=d[k].tolist(),mean=float(d[k].mean()),
                                 bootstrap_ci95=np.quantile(d[k][indices].mean(1),[.025,.975]).tolist()) for k in KEYS}
        tol=np.array([.001,1e-9,1e-9])
        good=int((np.column_stack([d[k] for k in KEYS[:3]])<=tol).all(1).sum())
        tails=all(d[k].mean()<=(1e-8 if 'excess' in k else 1e-9) for k in KEYS[4:])
        fc=sum(x['val_final']['cost']<=12 for x in candidate)
        fr=sum(x['val_final']['cost']<=12 for x in control)
        passed[ref]=dict(passed=all(d[k].mean()<0 for k in KEYS[:3]) and good>=4 and tails and fc>=fr,
                         primary_nonworsening_seeds=good,tail_means_nonworsening=tails,
                         feasible_seeds_candidate=fc,feasible_seeds_reference=fr)
    text=['# e58 연속 NH4 초과 크기 보조망 결과','',
          'e50 환경, 원 prior on, 세 조건 모두 CPU1 수집/GPU1 갱신, 각각5seed. 마지막 정책, 개발40일 x 모델5구성원. 현장 또는 독립 test 결과가 아니다.','',
          '| 조건 | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 | NH4 초과 mg h/L |', '| --- | --- | --- | --- | --- | --- |']
    for m,z in means.items():
        text.append('| '+m+' | '+' | '.join(f'{z[k]:.3f} ± {sd[m][k]:.3f}' for k in KEYS[:5])+' |')
    for ref,x in passed.items():
        text+=['',f"severity vs {ref}: 사전 동시 개선 판정 {x['passed']}, 세 지표 비악화 seed {x['primary_nonworsening_seeds']}/5, tail 비악화 {x['tail_means_nonworsening']}."]
    text+=['','## seed별 결과','', '| 조건 | seed | 공기량 | NH4 | DO | 초과 적분 | 최악 날짜 NH4 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for m,rs in rows.items():
        for r in rs:
            z=r['val_final'];text.append('| '+m+f" | {r['seed']} | "+' | '.join(f'{z[k]:.3f}' for k in ['air_m3','nh4_cost','do_cost','nh4_excess_mg_h_L','worst_day_nh4_cost'])+' |')
    text+=['','paired seed 차이와 탐색적 bootstrap95% 구간은 summary.json에 모두 기록한다. 반복 사용한 개발 자료와5seed로 신규성이나 현장 안전을 확증하지 않는다.']
    (root/'results.md').write_text('\n'.join(text)+'\n')
    same=[a['policy_fingerprint']==b['policy_fingerprint'] for a,b in zip(rows['baseline'],rows['detached'])]
    result=dict(means=means,sd=sd,comparisons=passed,paired_differences=differences,
                baseline_detached_final_policy_exact=same,rows=rows,test_evaluated=False)
    (root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    report(p.parse_args().root)
