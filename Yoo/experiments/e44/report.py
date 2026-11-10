"""모든 정책 완료 뒤 저장 경로를 독립 집계하고 실패를 포함해 표로 남긴다."""
import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY = ('air_m3','nh4_cost','do_cost')
GUARDS = ('nh4_excess_mg_h_L','worst_day_nh4_cost','worst_day_do_cost')


def main(root):
    manifest = json.loads((root/'manifest.json').read_text())
    results = {}
    lines = ['# e44 결과', '', '개발 val40일, 환경 구성원5개. 수치는 모델 내부 결과이며 실제 공정 성과가 아니다.', '',
             '| 방법 | 입력 | seed | 공기 m³/23.75h | NH4 비용 | DO 비용 | NH4 초과 적분 | 두 평균6/6 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    checks = 0
    for method in manifest['methods']:
        for mode in manifest['observation_modes']:
            for seed in manifest['seeds']:
                path = root/'runs'/method/mode/f'seed_{seed}'
                item = json.loads((path/'metrics.json').read_text())
                assert item['steps'] == manifest['steps_per_run']
                z = np.load(path/'val_rollout.npz')
                states,air = z['states'],z['air']
                assert states.shape == (200,95,5) and air.shape == (200,95)
                assert np.isfinite(states).all() and np.isfinite(air).all()
                n=(states[:,:,0]>4).sum(1)
                d=(states[:,:,2:].max(2)>2).sum(1)
                ex=np.maximum(states[:,:,0]-4,0).sum(1)*.25
                values={'air_m3':air.sum(1).mean()*.25,'nh4_cost':n.mean(),'do_cost':d.mean(),
                        'nh4_excess_mg_h_L':ex.mean(),'worst_day_nh4_cost':n.reshape(40,5).mean(1).max(),
                        'worst_day_do_cost':d.reshape(40,5).mean(1).max()}
                stored=item['val_final']
                for key,value in values.items():
                    assert np.isclose(stored[key],value,rtol=1e-10,atol=1e-10)
                    checks+=1
                rows=json.loads((path/'val_episodes.json').read_text())
                for key,v in [('air_m3',air.sum(1)*.25),('nh4_cost',n),('do_cost',d),('nh4_excess_mg_h_L',ex)]:
                    np.testing.assert_allclose([r[key] for r in rows],v,rtol=1e-10,atol=1e-10)
                    checks+=1
                assert len({r['day'] for r in rows})==40
                results[f'{method}/{mode}/{seed}']=stored
                lines.append(f"| {method} | {mode} | {seed} | {stored['air_m3']:.2f} | {stored['nh4_cost']:.3f} | {stored['do_cost']:.3f} | {stored['nh4_excess_mg_h_L']:.3f} | {stored['both_mean_limits']} |")
    pairs=[(f'{m}/previous',f'{m}/padded','input') for m in manifest['methods']]
    pairs += [(f'fcsrl_actor/{o}',f'fcsrl/{o}','extension') for o in manifest['observation_modes']]
    comparisons=[]
    lines += ['', '## 사전 지정 대조', '', '차이는 후보에서 기준을 뺀 값이다. 감소가 좋다. 2seed 탐색이며 통계적 확증이나 채택 결정이 아니다.', '',
              '| 후보 / 기준 | seed | 공기 차이 | NH4 차이 | DO 차이 | 탐색 기준 |',
              '| --- | --- | --- | --- | --- | --- |']
    for candidate,base,kind in pairs:
        for seed in manifest['seeds']:
            a,b=results[f'{candidate}/{seed}'],results[f'{base}/{seed}']
            delta={k:a[k]-b[k] for k in PRIMARY+GUARDS}
            passed=all(delta[k]<0 for k in PRIMARY) and all(delta[k]<=0 for k in GUARDS) and a['both_mean_limits']
            comparisons.append({'candidate':candidate,'base':base,'kind':kind,'seed':seed,'delta':delta,'exploratory_screen':passed})
            lines.append(f"| {candidate} / {base} | {seed} | {delta['air_m3']:.2f} | {delta['nh4_cost']:.3f} | {delta['do_cost']:.3f} | {passed} |")
    lines += ['', '탐색 기준은 세 주 지표 모두 감소, 초과 적분/최악 날짜 두 비용 악화 없음, 후보 평균6/6 만족이다. 두 seed 모두 통과해도 추가 검증 후보일 뿐이다. 모델 오차와 반복 사용한 개발 날짜의 한계는 남는다.']
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    (root/'comparison.json').write_text(json.dumps({'independent_checks':checks,'results':results,'comparisons':comparisons,
                                                  'adopted':False,'test_evaluated':False},indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print('REPORT COMPLETE',checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
