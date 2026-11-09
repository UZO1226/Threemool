"""DARROW 온라인 자료 기반 폭기 제어 Safe RL 환경 (벡터화).

- 한 에피소드는 실제 운전 하루(15분 x 96 행, 전이 95개)다. 유입량, 수온, 시각, 유입 성분은 그날 기록을 그대로 재생한다
- 행동은 총 공기량 하나다. [-1, 1] 을 [0, AIR_MAX] Nm3/h 로 바꾸고, 구간별 배분은 그 시각에 기록된 운전의 비율을 쓴다
  (기록된 총량이 0 이면 학습 구간 중앙값 비율). 즉 정책은 "얼마나"를, 기존 운전은 "어느 구간에"를 정한다
- 상태 전이는 단조 제약 대리 모델(surrogate_dw.GBMDynamics)이 계산한다. 에피소드마다 앙상블 구성원 하나를 고른다
- 보상은 -공기량 / AIR_MAX, 비용은 스텝마다 NH4 > NH4_MAX 이면 1, 구간 DO 의 최댓값 > DO_MAX 이면 1 을 더한다 (Yoo 의 환경과 같은 정의)
- 여러 환경을 한 번에 계산한다. 대리 모델 호출이 느려서 환경마다 따로 부르면 학습이 너무 오래 걸린다
"""
import numpy as np
import pandas as pd

from surrogate_dw import EXO, STATE, ZONES, GBMDynamics

AIR_MAX = 9000.0  # Nm3/h. 기록된 총 공기량의 99% 분위수 7,772, 최댓값 8,479 보다 조금 크게
NH4_MAX = 4.0  # mg N/L
DO_MAX = 2.0  # mg O2/L
OBS_DIM = 11
T_STEPS = 95  # 하루 96 행에서 전이 95개


class VecAerationEnv:
    def __init__(self, data_path, dynamics_path, split='train', n_env=8, seed=0, dynamics=None):
        d = pd.read_pickle(data_path)
        self.table, self.days = d['data'], d['days'][split]
        self.fractions = np.array([d['zone_fractions'][z] for z in ZONES])
        self.dyn = dynamics or GBMDynamics(dynamics_path)
        self.n, self.rng = n_env, np.random.default_rng(seed)
        self.day_arrays = {}
        for day in self.days:
            g = self.table.loc[day]
            z = g[ZONES].to_numpy(float)
            tot = z.sum(axis=1, keepdims=True)
            frac = np.where(tot > 1.0, z / np.maximum(tot, 1.0), self.fractions[None, :])
            self.day_arrays[day] = {'state': g[STATE].to_numpy(float), 'exo': g[EXO].to_numpy(float),
                                    'frac': frac, 'air': g.air.to_numpy(float), 'zone': z}
        self.day = [None] * n_env
        self.member = np.zeros(n_env, dtype=int)
        self.t = np.zeros(n_env, dtype=int)
        self.state = np.zeros((n_env, len(STATE)))
        self.prev_air = np.zeros(n_env)
        self.prev_zone = np.zeros((n_env, len(ZONES)))
        self.fixed = None  # 평가용: [(날짜, 구성원), ...] 를 순서대로 돌린다
        self._k = 0
        # 학습 속도: GBM 호출이 느려서 서로 다른 구성원을 섞으면 스텝마다 호출이 구성원 수만큼 늘어난다.
        # 학습에서는 갱신 한 번 동안 모든 환경이 같은 구성원을 쓰고, 갱신마다 구성원을 새로 뽑는다.
        self.fix_member = None

    def _pick(self, i):
        if self.fixed is not None:
            day, member = self.fixed[self._k % len(self.fixed)]
            self._k += 1
        else:
            day = self.days[self.rng.integers(len(self.days))]
            member = int(self.rng.integers(self.dyn.n_members)) if self.fix_member is None else int(self.fix_member)
        a = self.day_arrays[day]
        self.day[i], self.member[i], self.t[i] = day, member, 0
        self.state[i] = a['state'][0]
        self.prev_air[i] = a['air'][0]
        self.prev_zone[i] = a['zone'][0]

    def _obs(self):
        out = np.zeros((self.n, OBS_DIM), np.float32)
        for i in range(self.n):
            a = self.day_arrays[self.day[i]]
            exo, t = a['exo'][self.t[i]], self.t[i]
            out[i] = [*self.state[i], exo[0] / 1000.0, exo[1] / 10.0, exo[2], exo[3], exo[4] / 50.0,
                      self.prev_air[i] / AIR_MAX]
        return out

    def set_member(self, member):
        """진행 중인 모든 환경의 구성원을 바꾼다 (다음 reset 부터가 아니라 즉시)."""
        self.fix_member = int(member)
        self.member[:] = self.fix_member

    def reset(self):
        for i in range(self.n):
            self._pick(i)
        return self._obs()

    def step(self, action):
        """action (N,) in [-1, 1]. 반환: obs(자동 reset 뒤), reward, cost, truncated, terminal_obs, info"""
        a = np.clip(np.asarray(action, float).reshape(self.n), -1, 1)
        air = (a + 1) / 2 * AIR_MAX
        exo = np.stack([self.day_arrays[self.day[i]]['exo'][self.t[i]] for i in range(self.n)])
        frac = np.stack([self.day_arrays[self.day[i]]['frac'][self.t[i]] for i in range(self.n)])
        zone = air[:, None] * frac
        new = self.dyn.step(self.state, exo, zone, self.prev_zone, self.member)
        self.state, self.prev_air, self.prev_zone = new, air, zone
        self.t += 1
        v_nh4 = (new[:, 0] > NH4_MAX).astype(float)
        v_do = (new[:, 2:].max(axis=1) > DO_MAX).astype(float)
        trunc = self.t >= T_STEPS
        terminal_obs = self._obs()
        info = {'air': air, 'v_nh4': v_nh4, 'v_do': v_do, 'done_day': [self.day[i] for i in range(self.n)]}
        for i in np.flatnonzero(trunc):
            self._pick(i)
        return self._obs(), -air / AIR_MAX, v_nh4 + v_do, trunc, terminal_obs, info


def episodes(env):
    return [(d, m) for d in env.days for m in range(env.dyn.n_members)]


def batched_rollout(data_path, dynamics, split, act_batch, cost_limit, replay=False):
    """검증 또는 평가 구간의 모든 (날짜, 구성원) 에피소드를 한꺼번에 굴린다.
    act_batch(obs (N, 11)) -> (N,) in [-1, 1]. replay=True 면 기록된 총 공기량을 그대로 넣는다 (실제 운전 재생)."""
    eps_env = VecAerationEnv(data_path, None, split=split, n_env=1, dynamics=dynamics)
    eps = episodes(eps_env)
    env = VecAerationEnv(data_path, None, split=split, n_env=len(eps), dynamics=dynamics)
    env.fixed, env._k = eps, 0
    obs = env.reset()
    air = np.zeros(len(eps))
    cost = np.zeros(len(eps))
    nh4v = np.zeros(len(eps))
    dov = np.zeros(len(eps))
    for t in range(T_STEPS):
        if replay:
            rec = np.array([env.day_arrays[env.day[i]]['air'][env.t[i]] for i in range(env.n)])
            action = np.clip(rec / AIR_MAX * 2 - 1, -1, 1)
        else:
            action = act_batch(obs)
        obs, r, c, trunc, _, info = env.step(action)
        air += info['air']
        cost += c
        nh4v += info['v_nh4']
        dov += info['v_do']
        if t == T_STEPS - 1:
            assert trunc.all()
    rows = [{'day': d, 'member': m, 'air_m3': float(air[k] * 0.25), 'reward': float(-air[k] / AIR_MAX),
             'cost': float(cost[k]), 'nh4_viol_pct': float(100 * nh4v[k] / T_STEPS),
             'do_viol_pct': float(100 * dov[k] / T_STEPS)} for k, (d, m) in enumerate(eps)]
    s = {k: float(np.mean([r[k] for r in rows])) for k in ('air_m3', 'reward', 'cost', 'nh4_viol_pct', 'do_viol_pct')}
    s['days_within_limit_pct'] = float(100 * np.mean([r['cost'] <= cost_limit for r in rows]))
    s['n_episodes'] = len(rows)
    return s, rows
