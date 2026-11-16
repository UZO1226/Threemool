"""Park/e26 고르기 단계 (Park 작성). 학습 구간 마지막 40일에서 4 x 16 앙상블 조합 5개를 평가한다.

    python select_e26.py <이름> <구성원 폴더 이름>   (R, D 경로는 각자 환경에 맞게 바꾼다)
"""
import sys, json, itertools, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from darrow_env import BatchEnv, Dynamics, load_days, evaluate
from agent import FCSRL, ObsNormalizer
from min_brake import MinAirBrake, E15_MARGINS
from train_darrow import agent_act, darrow_config
D = '<Park/data/darrow>/'   # 각자 경로
R = '<결과>/'
splits, dd = load_days(D+'darrow_15min.pkl'); dyn = Dynamics(D+'darrow_surrogate_gbm.pkl'); env = BatchEnv(dd, dyn)
P1=[(u,m+0.1) for u,m in E15_MARGINS]
B19=dict(trim=True,margins=P1,do_limit=1.8,recover=(3,3.5),conflict=0.1)
name, d = sys.argv[1], sys.argv[2]
def load(p):
    ag=FCSRL(11,1,darrow_config(),ObsNormalizer(np.zeros((2,11)))); ag.load(p); return ag
ags = {s: load(f'{R}{d}/A_s{s}/model_final') for s in range(5)}
out = {}
for c in itertools.combinations(range(5), 4):
    act = lambda o, c=c: np.mean([np.asarray(agent_act(ags[s],o),float).reshape(len(o)) for s in c], axis=0)
    r = evaluate(env, splits['train'][-40:], act, shield=MinAirBrake(dyn,3.5,1,'others',**B19))
    out[''.join(map(str,c))] = r; print(name, c, r['air_m3'], r['cost'], flush=True)
json.dump(out, open(f'diag/scr26_{name}.json','w'))
