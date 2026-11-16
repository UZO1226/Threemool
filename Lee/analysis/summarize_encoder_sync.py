"""Measure logged encoder-sync KL before attributing failures to large EMA jumps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
for method in ('old_ppolag_fcsrl','new_ppolag_fcsrl'):
    for seed in range(5):
        folder=a.runs/method/f'seed_{seed}'
        config=json.loads((folder/'effective_config.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
        assert metrics['steps']==401280
        path=folder/'train_log.json';log=json.loads(path.read_text());assert len(log)==66
        values=np.array([r['encoder_sync_kl'] for r in log]);assert np.isfinite(values).all()
        rows.append({'method':method,'policy_seed':seed,'probe_kl_median':float(np.median(values)),
            'probe_kl_maximum':float(values.max()),'configured_ppo_target_kl':config['config']['target_kl'],
            'ema_tau':config['representation']['tau'],'final_validation_cost':metrics['val_final']['cost'],
            'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
result={'rows':rows,'test_evaluated':False,'policy_refit':False,
    'interpretation':'Logged analytic encoder-sync probe KL is small relative to the configured PPO sampled-action KL reference. Different estimators and a limited probe do not establish a global trust-region bound. These logs do not support a large single-update EMA shock as the demonstrated failure mechanism.'}
a.out.write_text(json.dumps(result,indent=2));print('Maximum logged sync KL:',max(r['probe_kl_maximum'] for r in rows),flush=True)
