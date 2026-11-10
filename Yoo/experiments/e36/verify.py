"""저장된 날짜별 결과를 독립 집계하고 범위 및 충족 판정을 검산한다."""
import json
from pathlib import Path
import numpy as np
import argparse
a=argparse.ArgumentParser()
a.add_argument('--out',type=Path,required=True)
p=a.parse_args().out
rows=json.loads((p/'rows.json').read_text())
count=0
for r in rows:
    f=p/f"{r['split']}_{r['environment']}_{r['name']}_episodes.json"
    if not f.exists():
        continue
    x=json.loads(f.read_text())
    assert len(x)==(940 if r['split']=='train' else 200)
    assert len({(i['day'],i['member']) for i in x})==len(x)
    for k in ('air_m3','nh4_cost','do_cost'):
        assert np.isclose(np.mean([i[k] for i in x]),r[k],rtol=0,atol=1e-7),(r['name'],k)
    assert r['both_mean_limits']==(r['nh4_cost']<=6 and r['do_cost']<=6)
    assert 0<=r['air_m3']<=9000*95*.25
    assert all(0<=i['nh4_cost']<=95 and 0<=i['do_cost']<=95 for i in x)
    count+=1
assert count == 40
result={'episode_aggregation_checks':count,'checks_passed':True}
(p/'verification.json').write_text(json.dumps(result,indent=2))
print(result)
