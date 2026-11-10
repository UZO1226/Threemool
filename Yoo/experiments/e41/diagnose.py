"""사후 설명: 정상/위반 구간의 오차 변화가 전체 평균에 미친 몫."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(out):
    z = np.load(out/'predictions.npz')
    result = {'posthoc_only': True, 'no_refit_or_threshold_change': True, 'splits': {}}
    for mode in ('holdout', 'cold', 'warm', 'one_step'):
        actual = z['one_step_actual'] if mode == 'one_step' else z['actual_'+mode]
        truth = np.broadcast_to(actual[:, None, :, 0], z['control_'+mode][..., 0].shape)
        data = {}
        for label, mask in (('normal', truth <= 4), ('violation', truth > 4)):
            rows = {'share_pct': float(mask.mean()*100), 'observed_mean_nh4': float(truth[mask].mean()), 'models': {}}
            for name in ('reference', 'control', 'augmented'):
                p = z[name+'_'+mode][..., 0]
                rows['models'][name] = {'mae': float(np.abs(p[mask]-truth[mask]).mean()),
                                         'bias': float((p[mask]-truth[mask]).mean()),
                                         'predicted_mean_nh4': float(p[mask].mean()),
                                         'predicted_exceed_pct': float((p[mask]>4).mean()*100)}
            delta = rows['models']['augmented']['mae'] - rows['models']['control']['mae']
            rows['augmented_minus_control_mae'] = delta
            rows['weighted_contribution_to_total_mae_change'] = float(mask.mean()*delta)
            data[label] = rows
        independent = float((np.abs(z['augmented_'+mode][..., 0]-truth)-np.abs(z['control_'+mode][..., 0]-truth)).mean())
        assert np.isclose(independent, sum(x['weighted_contribution_to_total_mae_change'] for x in data.values()))
        data['total_mae_change'] = independent
        result['splits'][mode] = data
    train = np.load(out/'training_examples.npz')
    actual_next = train['observed_x'][:, 0]+train['observed_y']
    result['fit_true_nh4_exceed_pct'] = float((actual_next>4).mean()*100)
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result['splits']['cold'],indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
