"""Park/e27 고르기 단계 (Park 작성). 학습 구간 마지막 40일에서 4 x 16 앙상블 조합 5개에 의견 차이 여유를 건다.

    python select_e27.py <A|B|C|D|E>   (D, R 경로는 각자 환경에 맞게 바꾼다. 기준은 Park/e26 select_e26.py base)
"""
import sys, json, itertools, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from darrow_env import BatchEnv, Dynamics, load_days, evaluate
from agent import FCSRL, ObsNormalizer
from min_brake import MinAirBrake, E15_MARGINS, EnsemblePolicy, DisagreementBrake
from train_darrow import agent_act, darrow_config
D = '<Park/data/darrow>/'   # 각자 경로
R = '<결과>/'
splits, dd = load_days(D+'darrow_15min.pkl'); dyn = Dynamics(D+'darrow_surrogate_gbm.pkl'); env = BatchEnv(dd, dyn)
P1=[(u,m+0.1) for u,m in E15_MARGINS]
B19=dict(trim=True,margins=P1,do_limit=1.8,recover=(3,3.5),conflict=0.1)
C={'A':(0.1,0.1),'B':(0.2,0.1),'C':(0.1,0.2),'D':(0,0.2),'E':(0.2,0)}
name=sys.argv[1]; up,down=C[name]
def load(p):
    ag=FCSRL(11,1,darrow_config(),ObsNormalizer(np.zeros((2,11)))); ag.load(p); return ag
ags={s: load(f'{R}dw_u16_runs/A_s{s}/model_final') for s in range(5)}
out={}
for c in itertools.combinations(range(5),4):
    pol=EnsemblePolicy([lambda o,g=ags[s]: agent_act(g,o) for s in c])
    r=evaluate(env, splits['train'][-40:], pol, shield=DisagreementBrake(MinAirBrake(dyn,3.5,1,'others',**B19), pol, 620, 1200, up, down))
    out[''.join(map(str,c))]=r; print(name,c,r['air_m3'],r['cost'],flush=True)
json.dump(out, open(f'diag/scr27_{name}.json','w'))
