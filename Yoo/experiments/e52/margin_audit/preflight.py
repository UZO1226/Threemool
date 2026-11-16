"""e53 실행 커밋, 공통 입력과 설정, 여유 0 smoke 재현을 확인한다."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RESULTS=Path('/home/jio/fcsrl_src/results')


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    e53=json.loads((RESULTS/'e53/main/manifest.json').read_text())
    old=json.loads((RESULTS/'e44/main/manifest.json').read_text())
    checks=[]
    for rel,value in old['source_sha256'].items():
        assert sha(ROOT/rel)==value,rel
        checks.append('e44_dependency:'+rel)
    for name,value in e53['source_sha256'].items():
        data=subprocess.check_output(['git','show',e53['code_commit']+':Yoo/experiments/e53/'+name],cwd=ROOT)
        assert hashlib.sha256(data).hexdigest()==value,name
        checks.append('e53_execution_commit:'+name)
    for name,p in [('data.pkl',RESULTS/'e30/prepared/data.pkl'),('prior.pt',RESULTS/'e30/prepared/prior.pt'),
                   ('corrected_surrogate.pkl',RESULTS/'e35/zone_fix/corrected_surrogate.pkl')]:
        assert sha(p)==e53['input_sha256'][name],name
        checks.append('input:'+name)
    for m in ['ppo','fcsrl']:
        for s in range(3):
            f=json.loads((RESULTS/f'e53/main/runs/{m}/d0.45/seed_{s}/effective_config.json').read_text())
            u=json.loads((RESULTS/f'e52/main/runs/{m}/k10/seed_{s}/effective_config.json').read_text())
            assert f.pop('delta')==.45 and u.pop('kappa')==10
            assert f==u,(m,s)
            checks.append(f'effective_config:{m}:{s}')
    smoke=json.loads((RESULTS/'e53/smoke/d0/train_log.json').read_text())
    baseline=json.loads((RESULTS/'e44/main/runs/fcsrl/padded/seed_0/train_log.json').read_text())[:2]
    assert len(smoke)==len(baseline)==2
    assert smoke==baseline
    checks.append('zero_margin_smoke_2_updates_exact')
    out=dict(checks=checks,count=len(checks),source_commit=e53['code_commit'],test_evaluated=False,weight_updates=0)
    (HERE/'preflight.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(checks))


if __name__=='__main__':main()
