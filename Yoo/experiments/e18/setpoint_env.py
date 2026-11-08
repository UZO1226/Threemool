"""e18 BSM1 DO 설정값 제어 환경

RL 은 5분마다 5번 조의 DO 설정값(0.5 ~ 3.0 mg/L)을 정하고, 원본 BSM1 의 DO PI(reginit.m 계수)가
1분마다 KLa5 를 계산해 그 설정값을 따라간다. 상위 설정값을 정하고 빠른 추종은 PI 가 맡는 구조다.
기존 e16 환경과 e17 비용 정의(유량 가중 NH4 부하)는 그대로 쓴다.

- 행동 a in [-1, 1] -> 설정값 0.5 + (a + 1) / 2 x 2.5. a = 0.2 가 기존 DO PI 의 설정값 2.0 이다
- PI 적분 상태는 하루 안에서 이어지고, 하루 시작 상태(snapshot)에 함께 저장해 reset 때 되돌린다
- 관측 12차원: e16 의 11개 중 마지막(직전 KLa)을 직전 설정값 / 3 으로 바꾸고, 현재 KLa / 240 을 덧붙인다
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e17'))
from bsm_mean import MeanEnv, QREF, episode_metrics, normalized_cost  # noqa: E402,F401
from bsm_env import BASE_KLAS, snapshot  # noqa: E402,F401

SP_LOW, SP_HIGH = 0.5, 3.0
OBS_DIM = 12


def action_to_setpoint(a):
    return SP_LOW + (float(np.clip(a, -1, 1)) + 1) / 2 * (SP_HIGH - SP_LOW)


def setpoint_to_action(sp):
    return (sp - SP_LOW) / (SP_HIGH - SP_LOW) * 2 - 1


class SetpointPI:
    """원본 reginit.m 의 DO PI (K=25, Ti=.002일, Tt=.001일, offset 144, 0 ~ 240 포화, anti-windup)
    를 1분마다 계산한다. e17 DOController 와 같고 설정값만 바꿀 수 있다."""

    def __init__(self, integral=0.):
        self.integral = float(integral)

    def __call__(self, do, sp):
        error = sp - do
        raw = 144. + 25. * error + self.integral
        output = float(np.clip(raw, 0., 240.))
        self.integral += (25. / .002 * error + (output - raw) / .001) / 1440
        return output


def snapshot_with_pi(model, pi, sp, pis=None):
    s = snapshot(model)
    s['pi_integral'] = np.array(pi.integral)
    s['setpoint'] = np.array(sp)
    for k, pk in (pis or {}).items():
        s[f'pi_integral_{k}'] = np.array(pk.integral)
    return s


class SetpointEnv(MeanEnv):
    """tanks: DO PI 로 제어하는 조 번호. 기본 (5,) 는 e18 본 시험 설정이고 3, 4번 조는 KLa 240 고정이다.
    (3, 4, 5) 이면 세 조 모두 같은 설정값을 각자의 PI 가 추종한다 (e18 원인 확인용)"""

    def __init__(self, state, weather='dry', start_day=0., days=1., tanks=(5,)):
        super().__init__(state, weather, start_day=start_day, days=days, controller=False)
        self.tanks = tuple(tanks)
        self.observation_space.shape  # noqa: B018
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

    def _obs(self):
        base = super()._obs()
        base[-1] = self.prev_sp / 3.
        return np.append(base, self.kla_now / 240.).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.pi = SetpointPI(float(self.initial.get('pi_integral', 0.)))
        self.pis = {k: SetpointPI(float(self.initial.get(f'pi_integral_{k}', 0.))) for k in self.tanks if k != 5}
        self.prev_sp = float(self.initial.get('setpoint', 2.))
        self.kla_now = 144. + float(self.pi.integral)
        self.kla_now = float(np.clip(self.kla_now, 0., 240.))
        obs, info = super().reset(seed=seed, options=options)
        return self._obs(), info

    def step(self, action):
        if self.t >= self.T:
            raise RuntimeError('reset is required')
        action = np.asarray(action)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError('Expected one finite action')
        sp = action_to_setpoint(action[0])
        begin = self.start_index + self.t * 5
        effluent, do, energy, mixing, klas5 = [], [], 0., 0., []
        for i in range(begin, begin + 5):
            kla = self.pi(float(self.model.y_out5[7]), sp)
            klas = BASE_KLAS.copy()
            klas[-1] = kla
            for k, pi_k in self.pis.items():
                klas[k - 1] = pi_k(float(getattr(self.model, f'y_out{k}')[7]), sp)
            self.model.step(i, klas)
            effluent.append(self.model.ys_eff.copy())
            do.append(float(self.model.y_out5[7]))
            klas5.append(kla)
            energy += self.model.ae / 60
            mixing += self.model.me / 60
        effluent = np.asarray(effluent)
        if not np.isfinite(effluent).all() or not np.isfinite([energy, *do]).all():
            raise FloatingPointError('Nonfinite BSM state')
        n, q = effluent[:, 9], effluent[:, 14]
        if np.min(n) < 0 or np.min(q) <= 0:
            raise FloatingPointError('Invalid concentration or discharge flow')
        self.t += 1
        self.prev_sp, self.kla_now, self.prev_kla = sp, kla, kla
        info = {'load': float(np.mean(n * q / (4 * QREF))), 'flow_ratio': float(q.mean() / QREF),
                'energy_kwh': float(energy), 'mixing_kwh': float(mixing), 'effluent': effluent,
                'do5': np.array(do), 'kla5': np.array(klas5), 'setpoint': sp,
                'violation_fraction': float((n > 4).mean())}
        return self._obs(), -energy / self.energy_reference, self.t == self.T, False, info


def evaluate(initial, policy, days_eval=7, tanks=(5,)):
    """e17 과 같은 평가: 조건화 상태에서 14일 연속 운전, 후반 7일 집계. 건기와 우기.
    policy(obs, env) -> 행동 [-1, 1]"""
    from ppo_tail import empirical_cvar

    result = {}
    for weather in ('dry', 'rain'):
        env = SetpointEnv(initial, weather=weather, days=14., tanks=tanks)
        obs, _ = env.reset()
        if hasattr(policy, 'reset'):
            policy.reset()
        infos = []
        for _ in range(env.T):
            obs, _, _, _, info = env.step(np.asarray(policy(obs, env), dtype=float).reshape(1))
            if env.t > (14 - days_eval) * 288:
                infos.append(info)
        week = episode_metrics(infos)
        week['setpoint_mean'] = float(np.mean([x['setpoint'] for x in infos]))
        days = [episode_metrics(infos[i:i + 288]) for i in range(0, len(infos), 288)]
        week['daily_nh4_cvar20'] = empirical_cvar([d['flow_weighted_nh4'] for d in days], .2)[1]
        result[weather] = {'week': week, 'days': days}
    return result


def training_days(states, policy, days=range(7), tanks=(5,)):
    """학습과 같은 조건(건기 d 일, 그날 시작 상태에서 하루 reset)으로 정책을 돌린다.
    states[d]: d 일 시작 snapshot. 반환: 날짜별 지표와 평균"""
    rows = []
    for d in days:
        env = SetpointEnv(states[d], weather='dry', start_day=float(d), days=1., tanks=tanks)
        obs, _ = env.reset()
        if hasattr(policy, 'reset'):
            policy.reset()
        infos = []
        for _ in range(env.T):
            obs, _, _, _, info = env.step(np.asarray(policy(obs, env), dtype=float).reshape(1))
            infos.append(info)
        m = episode_metrics(infos)
        rows.append({'day': int(d), 'flow_weighted_nh4': m['flow_weighted_nh4'],
                     'aeration_kwh_per_day': m['aeration_kwh_per_day'],
                     'nh4_violation_percent': m['nh4_violation_percent'], 'cost': m['cost']})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'day'}
    return {'days': rows, 'mean': mean}
