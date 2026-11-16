"""Park/e33 고르기 단계 (Park 작성). 학습 구간 마지막 40일에서 4 x 16 앙상블 조합 5개에 정책 공기량 조절 층을 건다.

    python select_e33.py <U90|S35_85|S3_85|S3_75|S25_70>   (D, R 경로는 각자 환경에 맞게 바꾼다. 기준은 Park/e30 select_e30.py P2)
"""
import sys, json, itertools, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from darrow_env import BatchEnv, Dynamics, load_days, evaluate
from agent import FCSRL, ObsNormalizer
from min_brake import MinAirBrake, E15_MARGINS, EnsemblePolicy, ScaledPolicy
from train_darrow import agent_act, darrow_config
D = '<Park/data/darrow>/'   # 각자 경로
R = '<결과>/'
# (factor, below)
C = {'U90': (0.9, float('inf')), 'S35_85': (0.85, 3.5), 'S3_85': (0.85, 3.0), 'S3_75': (0.75, 3.0), 'S25_70': (0.7, 2.5)}
splits, dd = load_days(D+'darrow_15min.pkl'); dyn = Dynamics(D+'darrow_surrogate_gbm.pkl'); env = BatchEnv(dd, dyn)
P1=[(u,m+0.1) for u,m in E15_MARGINS]
B19=dict(trim=True,margins=P1,do_limit=1.8,recover=(3,3.5),conflict=0.1)
name=sys.argv[1]; factor,below=C[name]
def load(p):
    ag=FCSRL(11,1,darrow_config(),ObsNormalizer(np.zeros((2,11)))); ag.load(p); return ag
ags={s: load(f'{R}dw_u16_runs/A_s{s}/model_final') for s in range(5)}
out={}
for c in itertools.combinations(range(5),4):
    pol=ScaledPolicy(EnsemblePolicy([lambda o,g=ags[s]: agent_act(g,o) for s in c]), factor, below)
    r=evaluate(env, splits['train'][-40:], pol, shield=MinAirBrake(dyn,3.5,1,'others',plan=(4,1.8,500.0),**B19))
    out[''.join(map(str,c))]=r; print(name,c,r['air_m3'],r['cost'],flush=True)
json.dump(out, open(f'diag/scr33_{name}.json','w'))
