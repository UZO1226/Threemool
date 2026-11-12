"""Summarize all seeds and paired uncertainty without checkpoint retuning."""
import argparse
import json
from pathlib import Path

import numpy as np

METHODS=['ppolag','ppolag_fcsrl','ppolag_fcsrl_curriculum']
KEYS=['air_m3','cost','nh4_viol_pct','do_viol_pct','days_within_limit_pct']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows={method:[json.loads((a.runs/method/f'seed_{seed}'/'metrics.json').read_text())
                  for seed in range(5)] for method in METHODS}
    manifest=json.loads((a.runs/'manifest.json').read_text())
    report={'manifest':manifest,'per_seed':rows,'summary':{},'paired_extension_minus_fcsrl':{}}
    for stage,key in [('final','val_final'),('selected','val_best')]:
        report['summary'][stage]={}
        for method,runs in rows.items():
            assert all(r['steps']==401280 for r in runs)
            assert [r['seed'] for r in runs]==list(range(5))
            report['summary'][stage][method]={k:{'mean':float(np.mean([r[key][k] for r in runs])),
                'sd':float(np.std([r[key][k] for r in runs]))} for k in KEYS}
        differences={}
        rng=np.random.default_rng(2026)
        indices=rng.integers(0,5,size=(10000,5))
        for k in KEYS:
            diff=np.array([r[key][k] for r in rows[METHODS[2]]])-np.array([r[key][k] for r in rows[METHODS[1]]])
            differences[k]={'values':diff.tolist(),'mean':float(diff.mean()),
                'paired_bootstrap_95_ci':np.quantile(diff[indices].mean(1),[.025,.975]).tolist()}
        report['paired_extension_minus_fcsrl'][stage]=differences
    final=report['summary']['final']
    diff=report['paired_extension_minus_fcsrl']['final']
    air_ratio=final[METHODS[2]]['air_m3']['mean']/final[METHODS[1]]['air_m3']['mean']
    mean_gate=diff['cost']['mean']<0 and diff['nh4_viol_pct']['mean']<0 and air_ratio<=1.05 and diff['do_viol_pct']['mean']<=1
    report['decision']={'mean_criteria_met':bool(mean_gate),'air_change_pct':float((air_ratio-1)*100),
        'cost_and_nh4_intervals_below_zero':bool(all(diff[k]['paired_bootstrap_95_ci'][1]<0 for k in ['cost','nh4_viol_pct'])),
        'test_evaluated':False,'exploratory':True}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Lee/e15 results','','All values are validation simulation results. Test remains reserved for team comparison.',
        'Air volume covers 95 intervals (23.75 hours) and is not measured electricity.','',
        '| Method | Air (Nm3) | Cost | NH4 violation (%) | DO violation (%) |',
        '| --- | --- | --- | --- | --- |']
    for method in METHODS:
        vals=[f"{final[method][k]['mean']:.3f} +/- {final[method][k]['sd']:.3f}" for k in KEYS[:4]]
        lines.append('| '+method+' | '+' | '.join(vals)+' |')
    lines+=['','## Paired extension minus FCSRL','']
    for k in KEYS[:4]:
        ci=diff[k]['paired_bootstrap_95_ci']
        lines.append(f"- {k}: {diff[k]['mean']:.4f}, paired bootstrap 95 percent interval [{ci[0]:.4f}, {ci[1]:.4f}].")
    lines+=['',f"Mean criteria met: {mean_gate}. Air change: {(air_ratio-1)*100:.2f} percent.",
        f"Both cost and NH4 intervals below zero: {report['decision']['cost_and_nh4_intervals_below_zero']}.",
        '', 'Five seeds and a shared imperfect simulator cannot establish real-plant improvement.',
        'Validation was used for environment selection, so these are exploratory comparisons.',
        'Selected-checkpoint secondary results and every seed are retained in summary.json.',
        f"Execution commit: {manifest['code_commit']}. See manifest source and artifact hashes."]
    (a.out/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
