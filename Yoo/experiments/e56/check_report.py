"""동일 결과와 교환관계를 전체 개선으로 표시하지 않는지 확인한다."""
import json
from pathlib import Path
import tempfile
import report


def run():
    p=json.loads((report.HERE/'protocol.json').read_text())
    with tempfile.TemporaryDirectory(prefix='e56_report_check_') as d:
        root=Path(d)
        for case in ['identical','tradeoff','joint']:
            out=root/case
            for method in p['methods']:
                for seed in p['seeds']:
                    baseline=json.loads((Path(p['baseline_audit'])/f'{method}_s{seed}'/'metrics.json').read_text())['summary']
                    z=dict(baseline)
                    if case!='identical':
                        for k in report.KEYS:z[k]*=.9
                    if case=='tradeoff':z['air_m3']=baseline['air_m3']*1.1
                    dest=out/'runs'/method/f'seed_{seed}';dest.mkdir(parents=True)
                    (dest/'metrics.json').write_text(json.dumps(dict(steps=401280,prior_mode='off',val_final=z)))
            report.report(out)
            r=json.loads((out/'summary.json').read_text())
            for method,z in r['methods'].items():
                assert z['exploratory_joint_screen']==(case=='joint'),(case,method)
                assert abs(z['air_mean_ratio_change_pct']-{'identical':0,'tradeoff':10,'joint':-10}[case])<1e-10
        print('PASS: identical/tradeoff rejected; joint screen and mean ratios correct for both methods')


if __name__=='__main__':run()
