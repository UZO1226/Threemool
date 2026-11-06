"""폭기조 상태 전이 모델 (5분 한 스텝)

인과 순서를 구조로 강제한다. 공기량은 DO 만 직접 바꾸고, NH4 는 DO 를 거쳐서만 공기량의 영향을 받는다.
공정 운전에서 공기량이 부하가 클 때 커지는 상관(교란)을 NH4 모델이 그대로 배우지 않게 하려는 것이다.

    DO 모델:  do[t+1]  - do[t]  = f(do[t], air[t], 단계, 수온, nh4[t], 유입량, 시각)
    NH4 모델: nh4[t+1] - nh4[t] = g(nh4[t], do[t], do[t+1], 단계, 수온, 유입량, 시각)

각 모델은 MLP 5개 앙상블이다. 학습 뒤 numpy 가중치로 저장해 환경에서 빠르게 쓴다.

    python surrogate.py --data Yoo/data/avedore/aved_5min.pkl --out <저장 폴더>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AIR_SCALE = 1000.0  # Nm3/h 를 천 단위로
Q_SCALE = 1000.0


def time_feats(idx):
    h = (idx.hour + idx.minute / 60.0).to_numpy()
    return np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)


def phase_onehot(p):
    p = np.asarray(p).round().astype(int)
    return np.stack([(p == k).astype(float) for k in (0, 1, 2)], 1)


def do_inputs(do, air, phase_oh, temp, nh4, q_in, s, c):
    air, q_in = np.asarray(air, dtype=float), np.asarray(q_in, dtype=float)
    return np.column_stack([do, air / AIR_SCALE, phase_oh, temp, nh4, q_in / Q_SCALE, s, c])


def nh4_inputs(nh4, do, do_next, phase_oh, temp, q_in, s, c):
    q_in = np.asarray(q_in, dtype=float)
    return np.column_stack([nh4, do, do_next, phase_oh, temp, q_in / Q_SCALE, s, c])


def make_pairs(r: pd.DataFrame, days):
    """주어진 날들의 (t, t+1) 쌍. 두 시점이 모두 유효한 것만 쓴다."""
    mask = pd.Series(r.index.date, index=r.index).astype(str).isin(set(days)).to_numpy()
    x = r.copy()
    nxt = x.shift(-1)
    ok = mask & x[["nh4", "do", "air", "phase", "temp", "q_in"]].notna().all(1).to_numpy() \
        & nxt[["nh4", "do"]].notna().all(1).to_numpy()
    s, c = time_feats(x.index)
    ph = phase_onehot(x["phase"].fillna(0))
    Xd = do_inputs(x.do, x.air, ph, x.temp, x.nh4, x.q_in, s, c)[ok]
    yd = (nxt.do - x.do).to_numpy()[ok]
    Xn = nh4_inputs(x.nh4, x.do, nxt.do, ph, x.temp, x.q_in, s, c)[ok]
    yn = (nxt.nh4 - x.nh4).to_numpy()[ok]
    return Xd, yd, Xn, yn


class NumpyEnsemble:
    """저장된 가중치로 추론만 하는 MLP 앙상블. 입력 정규화와 출력 역정규화를 포함한다."""

    def __init__(self, params: dict):
        self.mu, self.sd = np.array(params["x_mu"]), np.array(params["x_sd"])
        self.ymu, self.ysd = params["y_mu"], params["y_sd"]
        self.layers = [[(np.array(W), np.array(b)) for W, b in m] for m in params["members"]]

    def predict(self, X, member=None):
        X = (np.atleast_2d(X) - self.mu) / self.sd
        outs = []
        members = self.layers if member is None else [self.layers[member]]
        for m in members:
            h = X
            for i, (W, b) in enumerate(m):
                h = h @ W + b
                if i < len(m) - 1:
                    h = np.maximum(h, 0.0)
            outs.append(h[:, 0])
        return np.stack(outs, 0) * self.ysd + self.ymu  # (members, N)


def train_ensemble(X, y, Xv, yv, n_members=5, hidden=64, epochs=60, seed=0, device="cpu"):
    import torch
    import torch.nn as nn

    mu, sd = X.mean(0), X.std(0) + 1e-6
    ymu, ysd = float(y.mean()), float(y.std() + 1e-6)
    tX = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    ty = torch.tensor((y - ymu) / ysd, dtype=torch.float32, device=device)
    vX = torch.tensor((Xv - mu) / sd, dtype=torch.float32, device=device)
    vy = torch.tensor((yv - ymu) / ysd, dtype=torch.float32, device=device)
    members, logs = [], []
    for k in range(n_members):
        torch.manual_seed(seed * 100 + k)
        net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                            nn.Linear(hidden, 1)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        g = torch.Generator(device="cpu").manual_seed(seed * 100 + k)
        best, best_state = 1e9, None
        for ep in range(epochs):
            perm = torch.randperm(len(tX), generator=g)
            for i in range(0, len(tX), 1024):
                b = perm[i:i + 1024]
                loss = ((net(tX[b]).squeeze(-1) - ty[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                vl = ((net(vX).squeeze(-1) - vy) ** 2).mean().item()
            if vl < best:
                best, best_state = vl, [p.detach().cpu().numpy().copy() for p in net.parameters()]
        ws = best_state
        members.append([(ws[i].T.tolist(), ws[i + 1].tolist()) for i in range(0, len(ws), 2)])
        logs.append(best)
    return {"x_mu": mu.tolist(), "x_sd": sd.tolist(), "y_mu": ymu, "y_sd": ysd, "members": members,
            "val_mse_norm": logs}


class Surrogate:
    def __init__(self, path: Path):
        p = json.loads(Path(path).read_text())
        self.do_model = NumpyEnsemble(p["do"])
        self.nh4_model = NumpyEnsemble(p["nh4"])

    def step(self, nh4, do, air, phase, temp, q_in, hour, member):
        """한 스텝 전이. member 는 앙상블 구성원 번호 (에피소드마다 하나를 고른다)"""
        s, c = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        ph = phase_onehot([phase])
        xd = do_inputs([do], [air], ph, [temp], [nh4], [q_in], [s], [c])
        do_n = max(0.0, do + float(self.do_model.predict(xd, member)[0, 0]))
        xn = nh4_inputs([nh4], [do], [do_n], ph, [temp], [q_in], [s], [c])
        nh4_n = max(0.0, nh4 + float(self.nh4_model.predict(xn, member)[0, 0]))
        return nh4_n, do_n


def rollout(sur: Surrogate, day: pd.DataFrame, air, member=None):
    """하루 open loop 시뮬레이션. member=None 이면 앙상블 평균 대신 0번부터 차례로 평균낸다."""
    members = range(len(sur.do_model.layers)) if member is None else [member]
    res = []
    for m in members:
        nh4, do = float(day.nh4.iloc[0]), float(day.do.iloc[0])
        tr = []
        for t in range(len(day)):
            row = day.iloc[t]
            tr.append((nh4, do))
            h = row.name.hour + row.name.minute / 60
            nh4, do = sur.step(nh4, do, air[t], row.phase, row.temp, row.q_in, h, m)
        res.append(tr)
    return np.array(res).mean(0)  # (T, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_pickle(a.data)
    r, days = d["data"], d["days"]
    Xd, yd, Xn, yn = make_pairs(r, days["train"])
    Xdv, ydv, Xnv, ynv = make_pairs(r, days["val"])
    print(f"학습 쌍 {len(yd)}, 검증 쌍 {len(ydv)}")
    params = {"do": train_ensemble(Xd, yd, Xdv, ydv, device=a.device),
              "nh4": train_ensemble(Xn, yn, Xnv, ynv, device=a.device, seed=1)}
    (a.out / "surrogate.json").write_text(json.dumps(params))

    # 평가: 1 스텝 R2, 하루 open loop 오차 (평가 구간, 실제 공기량 재생)
    sur = Surrogate(a.out / "surrogate.json")
    rep = {}
    for split in ("val", "test"):
        Xd_, yd_, Xn_, yn_ = make_pairs(r, days[split])
        pd_ = sur.do_model.predict(Xd_).mean(0)
        pn_ = sur.nh4_model.predict(Xn_).mean(0)
        r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        rep[split] = {"one_step_r2_delta_do": r2(yd_, pd_), "one_step_r2_delta_nh4": r2(yn_, pn_)}
        errs = []
        for day in days[split]:
            g = r.loc[day]
            sim = rollout(sur, g, g.air.to_numpy())
            errs.append({
                "rmse_nh4": float(np.sqrt(np.mean((sim[:, 0] - g.nh4.to_numpy()) ** 2))),
                "rmse_do": float(np.sqrt(np.mean((sim[:, 1] - g.do.to_numpy()) ** 2))),
                "mean_nh4_sim": float(sim[:, 0].mean()), "mean_nh4_real": float(g.nh4.mean()),
                "mean_do_sim": float(sim[:, 1].mean()), "mean_do_real": float(g.do.mean()),
                "rmse_nh4_const": float(np.sqrt(np.mean((g.nh4.iloc[0] - g.nh4.to_numpy()) ** 2))),
            })
        e = pd.DataFrame(errs)
        rep[split].update({k: float(v) for k, v in e.mean().items()})
        rep[split]["corr_daily_mean_nh4"] = float(np.corrcoef(e.mean_nh4_sim, e.mean_nh4_real)[0, 1])
    (a.out / "surrogate_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
