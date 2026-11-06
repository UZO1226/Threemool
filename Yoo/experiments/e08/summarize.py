"""경계 전 동일성과 이후 표현 고정을 확인하고 seed별 결과를 집계한다."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/wwtp'))
import torch
from agents import load_agent
from env import AerationEnv
from evaluate import batched_rollout, episodes
from ppo_tail import risk_metrics


def read(path):
    return json.loads(path.read_text())


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def compare_models(a, b, representation_only=False):
    names = ('representation.pt',) if representation_only else ('model.pt', 'representation.pt')
    for name in names:
        assert same(torch.load(a/name, map_location='cpu', weights_only=True),
                    torch.load(b/name, map_location='cpu', weights_only=True)), (a, b, name)
    if not representation_only:
        assert read(a/'normalizer.json') == read(b/'normalizer.json')


def summarize(root, out):
    torch.set_num_threads(1)
    status, manifest = read(root/'status.json'), read(root/'manifest.json')
    assert status['phase'] == 'complete' and len(status['completed']) == 4
    assert all(r['exit_code'] == 0 for r in status['completed'])
    source = Path(__file__).resolve().parents[3]
    for relative, expected in manifest['source'].items():
        assert hashlib.sha256((source/relative).read_bytes()).hexdigest() == expected, relative
    for path, expected in manifest['inputs'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    results = read(root/'results.json')
    assert {(r['variant'], r['seed']) for r in results} == {(v,s) for v in ('continuous','freeze_half') for s in (0,1)}
    audits = []
    for seed in (0, 1):
        continuous, frozen = [root/'runs'/v/f'seed_{seed}' for v in ('continuous','freeze_half')]
        compare_models(continuous/'model_boundary', frozen/'model_boundary')
        assert read(continuous/'boundary.json') == read(frozen/'boundary.json')
        compare_models(frozen/'model_boundary', frozen/'model_final', representation_only=True)
        baseline = root.parents[1]/'e06/pilot/runs/ppolag_fcsrl'/f'seed_{seed}'
        compare_models(continuous/'model_final', baseline/'model_final')
        logs = [read(p/'train_log.json') for p in (continuous, frozen)]
        for a,b in zip(logs[0][:22], logs[1][:22]):
            assert {k:v for k,v in a.items() if k != 'representation_frozen_next'} == {
                k:v for k,v in b.items() if k != 'representation_frozen_next'}
        assert read(continuous/'pid_trace.json')[:22] == read(frozen/'pid_trace.json')[:22]
        assert len(logs[0]) == len(logs[1]) == 44
        assert all(r['representation_frozen_next'] == 0 for r in logs[0])
        assert all(r['representation_frozen_next'] == int(i >= 21) for i,r in enumerate(logs[1]))
        assert all('encoder_pred_feasi' not in r for r in logs[1][22:])
        for variant, run, log in zip(('continuous','freeze_half'), (continuous,frozen), logs):
            r = read(run/'metrics.json')
            assert r['steps'] == 404096 and r['evaluation_split'] == 'val'
            assert r['validation']['episodes'] == 215 and len([p for p in log if 'val' in p]) == 5
            effective = read(run/'effective_config.json')
            env = AerationEnv(effective['data'],effective['surrogate'],split='val')
            env.fixed = episodes(env)
            for kind in ('best','final'):
                agent = load_agent(run,env,cudaid=-1,checkpoint=kind)
                actual, rows = batched_rollout(env,env.sur,lambda o:agent.act_det(o)[:,0],15)
                actual.update(risk_metrics(rows))
                for key,value in actual.items():
                    assert abs(value-r[f'val_{kind}'][key])<1e-10, (variant,seed,kind,key)
            audits.append({'variant':variant,'seed':seed,'updates':len(log),
                           'maximum_lambda':max(p['lambda'] for p in log),
                           'restored_best_and_final_match':True})
    summary = {'launch_commit':manifest['commit'],'completed_at':status['updated_at'],
        'prefix_all_weights_normalizer_rng_log_pid_match':True,
        'frozen_representation_unchanged':True,'continuous_final_weights_match_e06':True,
        'audits':audits,'methods':{},'paired_deltas':{}}
    lines = ['# Yoo/e08 학습 중간 표현 고정 대조 결과','',
        'seed 2개, 각 404,096전이, 검증 43일 x 구성원 5개. 평균 ± 표준편차는 학습 seed 사이의 값이다. test는 사용하지 않았다.', '',
        'freeze_half는 22회 갱신(202,048전이) 뒤 표현 가중치를 고정하고 정책 및 가치망 학습은 계속했다. 선택 시점은 검증 결과로 조정하지 않았다.', '']
    labels = {'continuous':'계속 표현 학습','freeze_half':'중간부터 표현 고정'}
    for key,title in (('val_final','마지막 정책 (주요 비교)'),('val_best','동일 기준 선택 정책')):
        lines += [f'## {title}','', '| 방법 | 공기량 (Nm3/일) | 평균 비용 | CVaR20 | 평균 비용 15 이하 seed |',
                  '| --- | --- | --- | --- | --- |']
        for variant in labels:
            values = [r[key] for r in results if r['variant']==variant]
            stats = {'feasible_seeds':sum(v['cost']<=15 for v in values)}
            for field in ('air_m3','cost','cost_cvar'):
                stats[field+'_mean'] = statistics.mean(v[field] for v in values)
                stats[field+'_sd'] = statistics.stdev(v[field] for v in values)
            summary['methods'].setdefault(variant,{})[key]=stats
            lines.append(f'| {labels[variant]} | {stats["air_m3_mean"]:,.0f} ± {stats["air_m3_sd"]:,.0f} | '
                f'{stats["cost_mean"]:.3f} ± {stats["cost_sd"]:.3f} | {stats["cost_cvar_mean"]:.3f} ± '
                f'{stats["cost_cvar_sd"]:.3f} | {stats["feasible_seeds"]}/2 |')
        lines += ['', '| seed | 고정 후 공기량 차이 | 고정 후 평균 비용 차이 | 고정 후 CVaR20 차이 |',
                  '| --- | --- | --- | --- |']
        deltas = []
        for seed in (0,1):
            before = next(r[key] for r in results if r['variant']=='continuous' and r['seed']==seed)
            after = next(r[key] for r in results if r['variant']=='freeze_half' and r['seed']==seed)
            delta = {k:after[k]-before[k] for k in ('air_m3','cost','cost_cvar')}
            deltas.append({'seed':seed,**delta})
            lines.append(f'| {seed} | {delta["air_m3"]:+,.0f} | {delta["cost"]:+.3f} | {delta["cost_cvar"]:+.3f} |')
        summary['paired_deltas'][key]=deltas
        lines.append('')
    lines += ['## 개별 실행','','| 방법 | seed | 선택 epoch | 선택 비용 | 선택 공기량 | 마지막 비용 | 마지막 공기량 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for r in results:
        lines.append(f'| {labels[r["variant"]]} | {r["seed"]} | {r["best_epoch"]} | {r["val_best"]["cost"]:.3f} | '
            f'{r["val_best"]["air_m3"]:,.0f} | {r["val_final"]["cost"]:.3f} | {r["val_final"]["air_m3"]:,.0f} |')
    lines += ['', '## 점검 및 해석 범위', '',
        '모든 실행이 정상 종료했다. 같은 seed의 경계 전 저장 가중치, 관측 정규화, PyTorch 난수 상태, 학습 로그와 PID 기록이 일치했다. 고정군은 경계부터 마지막까지 모든 표현 가중치가 같았다. 계속 학습한 군의 마지막 가중치와 정규화는 e06과 정확히 같았다.', '',
        '선택 및 마지막 모델을 다시 읽어 고정 검증 결과가 정확히 같은지 확인했다. 선택은 평균 비용 한도를 만족한 시점 중 최고 보상 기준이다. 두 seed와 반복 사용한 검증 자료의 탐색 결과로 유의성이나 최종 수렴을 주장하지 않는다.', '',
        '이는 표현 갱신 중단의 대조이며 KL 보정의 효과를 시험한 것이 아니다. 대리 모델의 수질 과소예측도 해결하지 않았다.', '',
        f'실행 커밋 `{manifest["commit"]}`, 종료 UTC `{status["updated_at"]}`.', '',
        f'전체 로그와 모델: `{root}`.', '']
    out.mkdir(parents=True,exist_ok=True)
    (out/'results.md').write_text('\n'.join(lines))
    for name,value in (('summary',summary),('pilot_results',results),('launch',manifest)):
        (out/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False))
    print('\n'.join(lines))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    summarize(a.runs,a.out)
