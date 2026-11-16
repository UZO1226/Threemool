"""Park/e29 고르기 단계 (Park 작성). 학습 구간 마지막 40일에서 학습률을 섞은 4 x 16 앙상블 조합 5개를 평가한다.

    python select_e29.py <M1|M2|M3|M4>   (D, R 경로는 각자 환경에 맞게 바꾼다. 기준은 Park/e26 select_e26.py base)
"""
import sys, json, itertools, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from darrow_env import BatchEnv, Dynamics, load_days, evaluate
from agent import FCSRL, ObsNormalizer
from min_brake import MinAirBrake, E15_MARGINS, EnsemblePolicy
from train_darrow import agent_act, darrow_config
D = '<Park/data/darrow>/'   # 각자 경로
R = '<결과>/'
# 조합 (a, b, c, d) 의 자리별 학습률. 3e-4 는 Park/e22 dw_u16_runs, 나머지는 Park/e26 구성원
DIRS = {'3e-4': 'dw_u16_runs', '6e-4': 'dw_u16_lr6e-4', '1e-3': 'dw_u16_lr1e-3'}
MIX = {'M1': ('3e-4', '6e-4', '3e-4', '6e-4'), 'M2': ('3e-4', '1e-3', '3e-4', '1e-3'),
       'M3': ('3e-4', '6e-4', '3e-4', '1e-3'), 'M4': ('3e-4', '3e-4', '3e-4', '6e-4')}
splits, dd = load_days(D+'darrow_15min.pkl'); dyn = Dynamics(D+'darrow_surrogate_gbm.pkl'); env = BatchEnv(dd, dyn)
P1=[(u,m+0.1) for u,m in E15_MARGINS]
B19=dict(trim=True,margins=P1,do_limit=1.8,recover=(3,3.5),conflict=0.1)
name=sys.argv[1]
def load(p):
    ag=FCSRL(11,1,darrow_config(),ObsNormalizer(np.zeros((2,11)))); ag.load(p); return ag
ags={(lr,s): load(f'{R}{d}/A_s{s}/model_final') for lr,d in DIRS.items() for s in range(5)}
out={}
for c in itertools.combinations(range(5),4):
    pol=EnsemblePolicy([lambda o,g=ags[(lr,s)]: agent_act(g,o) for lr,s in zip(MIX[name],c)])
    r=evaluate(env, splits['train'][-40:], pol, shield=MinAirBrake(dyn,3.5,1,'others',**B19))
    out[''.join(map(str,c))]=r; print(name,c,r['air_m3'],r['cost'],flush=True)
json.dump(out, open(f'diag/scr29_{name}.json','w'))
