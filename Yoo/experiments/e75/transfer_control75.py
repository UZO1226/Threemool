"""Post hoc transfer null: replay original ABAC open loop on another member too."""
import argparse
import json
from pathlib import Path
import numpy as np
import plan75 as P


def main(root):
    ctx=P.context();controls={}
    for split in ('dev','val'):
        dates=P.select_dates(ctx[0][split]);eps=P.E.episodes(dates,ctx[2].n_members)
        acts,_=P.abac_actions(ctx,eps)
        source,_=P.simulate(ctx,eps,acts[:,None,:])
        target=[(d,(m+1)%ctx[2].n_members) for d,m in eps]
        op,_=P.simulate(ctx,target,acts[:,None,:])
        fbacts,_=P.abac_actions(ctx,target);fb,_=P.simulate(ctx,target,fbacts[:,None,:])
        reports=[json.loads(p.read_text()) for p in sorted((root/'jobs').glob(f'{split}_*/result.json'))]
        assert len(reports)==8
        sm=P.summary(source)
        for r in reports:
            for key in P.KEYS:
                np.testing.assert_allclose(sm[key],r['baseline'][key],rtol=0,atol=1e-8)
        controls[split]=dict(source_abac=sm,target_feedback_abac=P.summary(fb),target_open_loop_abac=P.summary(op),
                            candidates=[dict(mode=r['mode'],algorithm=r['algorithm'],seed=r['search_seed'],metrics=r['transfer']) for r in reports])
        np.savez_compressed(root/f'{split}_transfer_control.npz',source_commands=acts,
                            **{f'open_{k}':v for k,v in op.items()},**{f'feedback_{k}':v for k,v in fb.items()})
    P.write_json(root/'transfer_control.json',dict(post_hoc=True,controls=controls))
    lines=['# 사후 구성원 이전 대조','',
           '초기 이전 악화 결과를 본 뒤 추가했다. 원 ABAC 행동열도 고정해 다른 구성원에 넣는다. 후보를 재선택하지 않는다.','',
           '| 분할 | source ABAC 비용 | target 피드백 ABAC 비용 | target 고정 ABAC 비용 |',
           '| --- | --- | --- | --- |']
    for split,c in controls.items():
        lines.append(f"| {split} | {c['source_abac']['step_cost']:.3f} | {c['target_feedback_abac']['step_cost']:.3f} | {c['target_open_loop_abac']['step_cost']:.3f} |")
    lines+=['','| 분할/목적/탐색/난수 | 후보 이전 비용 | 고정 ABAC 이전 대비 비용 차이 |','| --- | --- | --- |']
    for split,c in controls.items():
        for r in c['candidates']:
            cost=r['metrics']['step_cost']
            lines.append(f"| {split}/{r['mode']}/{r['algorithm']}/{r['seed']} | {cost:.3f} | {cost-c['target_open_loop_abac']['step_cost']:+.3f} |")
    lines+=['','닫힌 고리 ABAC와 열린 고리 후보의 큰 차이만으로 후보에 특이적인 모델 착취를 주장하지 않는다. 같은 모델 계열에서 피드백 부재와 모델 차이가 함께 작용한 민감도 검사다.']
    (root/'transfer_control.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({s:{k:v for k,v in c.items() if k!='candidates'} for s,c in controls.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
