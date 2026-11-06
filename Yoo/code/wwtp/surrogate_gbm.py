"""평가 전용 대리 모델 (Yoo/e03)

학습용 대리 모델(surrogate.py, MLP 앙상블)로 학습하고 같은 모델로 평가하면, 정책이 그 모델의 오차를
이용한 만큼 결과가 낙관적으로 나온다. 그래서 성질이 다른 모델을 따로 만들어 평가에만 쓴다.

- 입력과 인과 구조는 학습용과 같다 (공기량은 DO 로만, NH4 는 DO 를 거쳐서만)
- 모델은 gradient boosting 나무 (sklearn HistGradientBoostingRegressor) 5개 앙상블이다.
  구성원마다 학습 쌍의 80% 를 무작위로 뽑고 난수 seed 를 달리한다
- 물리적으로 알려진 방향을 단조 제약으로 건다
  DO 변화량은 공기량에 대해 증가, NH4 변화량은 다음 스텝 DO 에 대해 감소
- 나무 모델은 입력이 도달한 잎의 값을 쓴다. MLP와 외삽 방식이 달라 민감도 확인에 쓸 수 있지만,
  같은 데이터와 입력 구조를 쓰므로 공통 오차나 정책의 모델 오차 의존성을 배제하지 못한다

    python surrogate_gbm.py --data Yoo/data/avedore/aved_5min.pkl --out <저장 폴더>
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import do_inputs, make_pairs, nh4_inputs, phase_onehot, report

# 입력 열 순서는 surrogate.do_inputs, nh4_inputs 와 같다
DO_MONO = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # do, air(+), 단계 3, 수온, nh4, 유입량, sin, cos
NH4_MONO = [0, 0, -1, 0, 0, 0, 0, 0, 0, 0]  # nh4, do, do_next(-), 단계 3, 수온, 유입량, sin, cos


def fit_members(X, y, mono, n_members=5, frac=0.8, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_members):
        idx = rng.choice(len(y), int(frac * len(y)), replace=False)
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, monotonic_cst=mono, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=seed * 10 + k)
        m.fit(X[idx], y[idx])
        out.append(m)
    return out


class _Members:
    def __init__(self, models):
        self.models = models

    def predict(self, X, member=None):
        X = np.atleast_2d(X)
        ms = self.models if member is None else [self.models[member]]
        return np.stack([m.predict(X) for m in ms], 0)  # (members, N)


class GBMSurrogate:
    """surrogate.Surrogate 와 같은 사용법 (step, step_all, n_members)"""

    def __init__(self, path):
        with open(path, "rb") as f:
            p = pickle.load(f)
        self.do_model, self.nh4_model = _Members(p["do"]), _Members(p["nh4"])

    @property
    def n_members(self):
        return len(self.do_model.models)

    def step(self, nh4, do, air, phase, temp, q_in, hour, member):
        s, c = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        ph = phase_onehot([phase])
        xd = do_inputs([do], [air], ph, [temp], [nh4], [q_in], [s], [c])
        do_n = max(0.0, do + float(self.do_model.predict(xd, member)[0, 0]))
        xn = nh4_inputs([nh4], [do], [do_n], ph, [temp], [q_in], [s], [c])
        nh4_n = max(0.0, nh4 + float(self.nh4_model.predict(xn, member)[0, 0]))
        return nh4_n, do_n

    def step_all(self, nh4, do, air, phase, temp, q_in, hour):
        res = [self.step(nh4, do, air, phase, temp, q_in, hour, m) for m in range(self.n_members)]
        return np.array([r[0] for r in res]), np.array([r[1] for r in res])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import sklearn

    a.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_pickle(a.data)
    r, days = d["data"], d["days"]
    Xd, yd, Xn, yn = make_pairs(r, days["train"])
    params = {"do": fit_members(Xd, yd, DO_MONO, seed=0), "nh4": fit_members(Xn, yn, NH4_MONO, seed=1),
              "sklearn": sklearn.__version__}
    path = a.out / "surrogate_gbm.pkl"
    with open(path, "wb") as f:
        pickle.dump(params, f)
    rep = report(GBMSurrogate(path), r, days)
    rep["sklearn"] = sklearn.__version__
    (a.out / "surrogate_gbm_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
