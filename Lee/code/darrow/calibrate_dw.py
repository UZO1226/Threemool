"""실제 운전 기록을 대리 모델에서 재생해 하루 비용을 재고, 비용 한도를 정한다 (학습 구간만 사용, 정책 학습 전).

한도 규칙: 학습 구간에서 실제 운전 재생의 하루 평균 비용을 반올림한 값. 즉 "실제 운전(대리 모델이 보기에)보다
위반이 많지 않으면서 공기량을 줄이는" 문제다. 검증과 평가 구간은 한도를 정하는 데 쓰지 않는다.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fcsrl_compat  # noqa: F401,E402
import numpy as np  # noqa: E402
from env_dw import batched_rollout  # noqa: E402
from surrogate_dw import GBMDynamics  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--surrogate', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
dyn = GBMDynamics(a.surrogate)
out = {}
for split in ('train', 'val', 'test'):
    s, _ = batched_rollout(a.data, dyn, split, None, 1e9, replay=True)
    out[split] = s
    print(f'{split}: 실제 운전 재생 공기량 {s["air_m3"]:.0f} Nm3/일, 하루 비용 {s["cost"]:.2f}, NH4 위반 {s["nh4_viol_pct"]:.2f}%, DO 위반 {s["do_viol_pct"]:.2f}%')
out['cost_limit'] = float(round(out['train']['cost']))
print('비용 한도(학습 구간 재생의 하루 평균 비용 반올림):', out['cost_limit'])
a.out.write_text(json.dumps(out, indent=2))
