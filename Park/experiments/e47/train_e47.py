"""Park/e47 학습 진입점 (Park 작성). train_darrow.py 의 main 을 그대로 부르되 제거 대조 방법 이름을 먼저 등록한다.

    python train_e47.py --method fcsrl_rand --seed <s> --data .. --surrogate .. --min-brake-target 3.5 --updates 16 --out <결과>/dw_rand16_runs/R_s<s>
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from fcsrl_ablation import register  # noqa: E402
import train_darrow  # noqa: E402

if __name__ == '__main__':
    register()
    train_darrow.main()
