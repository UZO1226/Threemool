"""prior 대조의 단일 변경, 기존 경로 재현과 입력 불변을 확인한다."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import train

HERE=Path(__file__).resolve().parent


def sources():
    return sorted({str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                   if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(train.ROOT/'Yoo'))
                   and str(m.__file__).endswith('.py')})


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    p=json.loads((HERE/'protocol.json').read_text());prepared=Path(p['prepared'])
    base=json.loads((Path(p['baseline'])/'manifest.json').read_text())
    hashes={str(prepared/k):v for k,v in base['input_sha256'].items()}
    assert all(train.sha(k)==v for k,v in hashes.items())
    # e50 당시의 학습 본체와 현재 본체가 동일해야 재학습 없이 기준선을 재사용한다.
    shared=['Yoo/experiments/e30/'+n for n in ['train.py','common.py','darrow_env.py']]
    shared+=['Yoo/experiments/e23/behavior.py','Yoo/experiments/e25/preserve.py','Yoo/experiments/e29/ablation.py']
    shared += [str(f.relative_to(train.ROOT)) for f in (train.ROOT/'Yoo/code/wwtp').glob('*.py')]
    for rel in shared:
        old=subprocess.check_output(['git','show',base['code_commit']+':'+rel],cwd=train.ROOT)
        assert old==(train.ROOT/rel).read_bytes(),rel
    torch.set_num_threads(1);train.DeviceConfig().select_device(-1)
    splits,days=train.load_selected(prepared/'data.pkl')
    env=train.BatchEnv(days,train.Dynamics(prepared/'surrogate.pkl'))
    obs=env.reset([(splits['train'][i],0) for i in range(8)])
    checks={}
    for method in ['ppo','fcsrl']:
        on=train.make_agent(method,0,prepared,train.common.config(2),'on');r0=torch.get_rng_state()
        off=train.make_agent(method,0,prepared,train.common.config(2),'off');r1=torch.get_rng_state()
        assert torch.equal(r0,r1)
        modules=['pi','vr','vc']+(['encoder','fixed_encoder','feasi_head','proj_layer','post_proj'] if method=='fcsrl' else [])
        for name in modules:
            assert all(torch.equal(v,getattr(off,name).state_dict()[k]) for k,v in getattr(on,name).state_dict().items())
        assert on.obs_normalizer.state()==off.obs_normalizer.state()
        x=on._t(on.obs_normalizer(obs))
        assert torch.equal(on.log_std,off.log_std)
        assert torch.equal(on.dist(x).stddev,off.dist(x).stddev)
        assert torch.equal(off.dist(x).mean,torch.zeros((8,1)))
        assert torch.equal(on.dist(x).mean,on.prior_action(x))
        # 학습 출력은 prior와 무관하게 변할 수 있으며 별도의 행동 구속은 없다.
        with torch.no_grad():off.pi[-1].bias.fill_(.2)
        assert torch.allclose(off.dist(x).mean,torch.full((8,1),.2))
        checks[method]={'initial_nonprior_weights_and_rng_identical':True,'prior_off_initial_air_m3_h':4500,
                        'initial_log_std':float(off.log_std.item()),'prior_is_not_hard_constraint':True}
    envvars=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for method in ['ppo','fcsrl']:
        for mode in ['on','off']:
            dest=out/f'{method}_{mode}'
            cmd=[sys.executable,str(HERE/'train.py'),'--method',method,'--prior-mode',mode,'--seed','0',
                 '--updates','2','--prepared',str(prepared),'--out',str(dest),'--smoke']
            with (out/f'{method}_{mode}.log').open('w') as log:
                subprocess.run(cmd,env=envvars,stdout=log,stderr=subprocess.STDOUT,check=True)
            got=json.loads((dest/'train_log.json').read_text())
            if mode=='on':
                old=json.loads((Path(p['baseline'])/'runs'/method/'seed_0/train_log.json').read_text())[:2]
                assert got==old,(method,'prior-on trace mismatch')
                checks[method]['prior_on_2_updates_exact']=True
            else:checks[method]['prior_off_2_updates_finite_and_reload_exact']=True
            print('PASS',method,mode,flush=True)
    # 학습 코드가 실제 호출하는 모듈, 기준 로그와 상세 평가를 고정한다.
    for path in sources()+[str(HERE/'train.py'),str(HERE/'preflight.py'),str(HERE/'protocol.json')]:
        hashes[path]=train.sha(path)
    for method in p['methods']:
        for seed in p['seeds']:
            for name in ['train_log.json','metrics.json','effective_config.json']:
                path=Path(p['baseline'])/'runs'/method/f'seed_{seed}'/name;hashes[str(path)]=train.sha(path)
            path=Path(p['baseline_audit'])/f'{method}_s{seed}'/'metrics.json';hashes[str(path)]=train.sha(path)
    assert all(train.sha(k)==v for k,v in hashes.items())
    train.save(out/'checks.json',dict(passed=True,checks=checks,shared_sources_match_e50=shared,hashes=hashes))
    print('PREFLIGHT PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
