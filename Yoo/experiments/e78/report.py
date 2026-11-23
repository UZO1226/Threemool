"""Audit per-episode metrics and report paired policy-seed outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np

QUALITY = ['nh', 'do', 'excess']
MODES = ['base', 'down', 'both']
TOL = 1e-8


def wins(d, extra):
    if not all(d[k] <= TOL for k in ['air', *QUALITY]):
        return False
    if extra:
        return d['nh'] < -TOL or d['excess'] < -TOL
    return d['air'] < -TOL


def audit(root, r):
    with np.load(root/f"{r['seed']}_{r['split']}_{r['mode']}.npz") as z:
        state, air, direction, raw = (z[k] for k in ['states','air','direction','raw'])
        n_days = 48 if r['split'] == 'train48' else 40
        assert state.shape == (n_days * 5,95,5) and len(z['dates']) == n_days
        assert (np.abs(air-raw-500*direction)<1e-7).all()
        nh = np.count_nonzero(state[:,:,0]>4,axis=1)
        do = (state[:,:,2:]>2).any(axis=2).sum(1)
        metrics = dict(air=air.sum(1)/4, nh=nh, do=do, excess=np.clip(state[:,:,0]-4,0,None).sum(1)/4,
                       up_steps=(direction>0).sum(1),down_steps=(direction<0).sum(1),raw_air=raw.sum(1)/4,
                       limit_pct=100*((nh+do)<=12), reward=-air.sum(1)/9000)
        for k,v in metrics.items():
            np.testing.assert_allclose(v,z['day_'+k],atol=TOL,rtol=0)
            assert abs(v.mean()-r[k]) < TOL


def report(root, dest):
    rows=json.loads((root/'records.json').read_text())
    assert len(rows)==30
    for seed in range(5):
        with np.load(root/f'fit{seed}.npz') as z:
            assert (z['keys'][:,0] < 140).all()
            lo, base, hi = z['low'], z['base'], z['high']
            cheap = (base[:,:,0]-lo[:,:,0] >= 25) & (lo[:,:,1:] <= base[:,:,1:]+TOL).all(2)
            useful = ((base[:,:,1]-hi[:,:,1] >= 1) | (base[:,:,3]-hi[:,:,3] >= .1))
            useful &= (hi[:,:,1:] <= base[:,:,1:]+TOL).all(2)
            labels = np.select([z['down'] & (cheap.sum(1)>=4), z['up'] & (useful.sum(1)>=4)], [-1,1], default=0)
            np.testing.assert_array_equal(labels,z['labels'])
            assert rows[6*seed]['fit_labels'] == {str(k):int((labels==k).sum()) for k in [-1,0,1]}
    for row in rows:
        audit(root,row)
    md=['# e78 하루 피드백 결과','',
        '각 정책 seed를 같은 날짜와 같은 GBM 구성원으로 비교했다. 공기는 m³/23.75h, NH4와 DO는 위반 스텝/일, NH4 초과 적분은 mg h/L다. 개입은 스텝/일이다. 원본은 base, 감량 전용은 down, 양방향은 both로 표시했다.','',
        '| 날짜 | 모드 | 공기 | NH4 | DO | NH4 초과 | 한도12 만족일 % | reward | 감량 | 증량 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split in ('train48','winter'):
        for mode in MODES:
            group=[r for r in rows if r['mode']==mode and r['split']==split]
            assert sorted(r['seed'] for r in group)==list(range(5))
            md.append('| '+split+' | '+mode+' | '+' | '.join(f"{np.mean([r[k] for r in group]):.3f}"
                for k in ['air','nh','do','excess','limit_pct','reward','down_steps','up_steps'])+' |')
    md += ['', '## 겨울 검증 짝 비교', '',
           '| 변경 | seed | 공기 차 | NH4 차 | DO 차 | 초과 적분 차 | 공동 개선 |',
           '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for ref,mode in [('base','down'),('base','both'),('down','both')]:
        ds=[]
        for seed in range(5):
            pair={r['mode']:r for r in rows if r['seed']==seed and r['split']=='winter'}
            d={k:pair[mode][k]-pair[ref][k] for k in ['air',*QUALITY]}
            ds.append(d)
            md.append(f'| {ref} → {mode} | {seed} | '+' | '.join(f'{v:+.3f}' for v in d.values())+
                      f" | {wins(d,ref=='down')} |")
        avg={k:np.mean([d[k] for d in ds]) for k in ds[0]}
        n=sum(wins(d,ref=='down') for d in ds)
        passed=n>=4 and wins(avg,ref=='down')
        md.append(f'| {ref} → {mode} | 평균 | '+' | '.join(f'{v:+.3f}' for v in avg.values())+
                  f' | {n}/5, {"충족" if passed else "미달"} |')
    md += ['', '직접 증량/감량의 공기 부피는 해당 개입 횟수 × 125 m³다. 나머지 공기 차이는 서로 다른 수질 상태에서 기본 정책이 낸 명령 차이를 포함한다. 모든 30개 궤적 파일에서 공기, 보상, 위반, NH4 초과 적분, 방향별 개입 횟수를 재계산했다.']
    md += ['', '## 증량 허용 시 공기 차이의 구성', '',
           '겨울 검증에서 양방향 실행 값에서 감량 전용 실행 값을 뺐다. 각 열은 23.75시간당 m³다. '
           '행동 경로가 달라지므로 이는 관측된 부피의 항등 분해다.', '',
           '| seed | 직접 증량 | 직접 감량 차이 | 각 경로의 기본 명령 차이 | 전체 차이 |',
           '| --- | ---: | ---: | ---: | ---: |']
    terms = []
    for seed in range(5):
        r = {x['mode']:x for x in rows if x['seed']==seed and x['split']=='winter'}
        down, both = r['down'], r['both']
        parts = [125*(both['up_steps']-down['up_steps']), -125*(both['down_steps']-down['down_steps']),
                 both['raw_air']-down['raw_air'], both['air']-down['air']]
        assert abs(sum(parts[:3])-parts[3]) < TOL
        terms.append(parts)
        md.append(f'| {seed} | '+' | '.join(f'{v:+.2f}' for v in parts)+' |')
    md.append('| 평균 | '+' | '.join(f'{v:+.2f}' for v in np.mean(terms,axis=0))+' |')
    (dest/'results.md').write_text('\n'.join(md)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args()
    report(a.root,a.dest)
