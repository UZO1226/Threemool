"""e17 평균 수질 비용과 유한한 하루 에피소드. 기존 환경 및 agent는 수정하지 않는다."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'e16'))
sys.path.insert(0, str(ROOT / 'Yoo/code/wwtp'))
from bsm_env import BSMAerationEnv, BASE_KLAS, snapshot, restore, make_plant

QREF = 18446.
STEPS_PER_DAY = 288


def normalized_cost(load, flow_ratio):
    """완결한 에피소드의 수질 부하를 유량으로 나눈다. 정책 관측에 미래값을 넣지 않는다."""
    load, flow_ratio = np.asarray(load), np.asarray(flow_ratio)
    if load.shape != flow_ratio.shape or not np.isfinite(load).all() or not np.isfinite(flow_ratio).all():
        raise ValueError('Expected matching finite load and flow arrays')
    if np.min(load) < 0 or np.min(flow_ratio) <= 0:
        raise ValueError('Negative load or nonpositive flow')
    return load / flow_ratio.mean(axis=0, keepdims=True)


class DOController:
    """원본 reginit.m의 DO PI 계수로 1분마다 계산하는 이상적 센서 기준 제어."""
    def __init__(self):
        self.integral = 0.

    def __call__(self, do):
        error = 2. - do
        raw = 144. + 25. * error + self.integral
        output = float(np.clip(raw, 0., 240.))
        self.integral += (25. / .002 * error + (output - raw) / .001) / 1440
        return output


class MeanEnv(BSMAerationEnv):
    def __init__(self, state, weather='dry', start_day=0., days=1., controller=False):
        super().__init__(state, weather, start_day=start_day, days=days)
        self.use_controller = controller
        self.controller = None
        from bsm2_python.bsm2.init import asm1init_bsm1 as init
        volumes = np.array([getattr(self.model, f'reactor{i}').volume for i in range(1, 6)])
        saturation = np.array([getattr(init, f'SOSAT{i}') for i in range(1, 6)])
        self.energy_reference = self.model.performance.aerationenergy_step(BASE_KLAS, volumes, saturation) * 5 / 60

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.controller = DOController() if self.use_controller else None
        return out

    def step(self, action):
        if self.t >= self.T:
            raise RuntimeError('reset is required')
        action = np.asarray(action)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError('Expected one finite action')
        kla = (float(np.clip(action[0], -1, 1)) + 1) * 120
        begin = self.start_index + self.t * 5
        effluent, do, energy, mixing, klas5 = [], [], 0., 0., []
        for i in range(begin, begin + 5):
            if self.controller is not None:
                kla = self.controller(float(self.model.y_out5[7]))
            klas = BASE_KLAS.copy()
            klas[-1] = kla
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
        self.prev_kla = kla
        info = {'load': float(np.mean(n * q / (4 * QREF))), 'flow_ratio': float(q.mean() / QREF),
                'energy_kwh': float(energy), 'mixing_kwh': float(mixing), 'effluent': effluent,
                'do5': np.array(do), 'kla5': np.array(klas5)}
        # 일별 reset은 유한 에피소드 종료다. 가치와 feasibility를 다음 날로 bootstrap하지 않는다.
        return self._obs(), -energy / self.energy_reference, self.t == self.T, False, info


def episode_metrics(infos):
    eff = np.concatenate([x['effluent'] for x in infos])
    q, n = eff[:, 14], eff[:, 9]
    do = np.concatenate([x['do5'] for x in infos])
    kla = np.concatenate([x['kla5'] for x in infos])
    from bsm2_python.bsm2.plantperformance import PlantPerformance
    from bsm2_python.bsm2.init.plantperformanceinit_bsm1 import PP_PAR
    advanced = PlantPerformance(PP_PAR).advanced_quantities(eff, ('totalN', 'COD', 'BOD5e', 'X_TSS'))
    mean_nh4 = float(np.average(n, weights=q))
    steps = len(infos)
    return {'flow_weighted_nh4': mean_nh4, 'cost_ratio': mean_nh4 / 4,
            'cost': mean_nh4 / 4 * steps, 'nh4_violation_percent': float((n > 4).mean() * 100),
            'aeration_kwh_per_day': sum(x['energy_kwh'] for x in infos) * 288 / steps,
            'mixing_kwh_per_day': sum(x['mixing_kwh'] for x in infos) * 288 / steps,
            'do5_mean': float(do.mean()), 'do5_above_target_percent': float((do > 2).mean() * 100),
            'kla5_mean': float(kla.mean()), 'days': steps / 288,
            'other_flow_weighted': dict(zip(('totalN', 'COD', 'BOD5e', 'TSS'),
                                            np.average(advanced, axis=0, weights=q).tolist()))}


def evaluate(initial, policy, controller=False):
    result = {}
    for weather in ('dry', 'rain'):
        env = MeanEnv(initial, weather=weather, days=14., controller=controller)
        obs, _ = env.reset()
        infos = []
        for _ in range(env.T):
            obs, _, _, _, info = env.step(np.asarray(policy(obs)).reshape(1))
            if env.t > 7 * 288:
                infos.append(info)
        week = episode_metrics(infos)
        days = [episode_metrics(infos[i:i + 288]) for i in range(0, len(infos), 288)]
        from ppo_tail import empirical_cvar
        week['daily_nh4_cvar20'] = empirical_cvar([d['flow_weighted_nh4'] for d in days], .2)[1]
        result[weather] = {'week': week, 'days': days}
    return result
