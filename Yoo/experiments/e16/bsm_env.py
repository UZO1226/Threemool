"""BSM1의 원본 공정식을 호출하는 별도 탐색 환경. 기존 Avedore 환경은 유지한다."""
from pathlib import Path
import io
import os
import zipfile

import bsm2_python
from bsm2_python import BSM1OL
import gymnasium as gym
import numpy as np

PACKAGE = Path(bsm2_python.__file__).parent
BASE_KLAS = np.array([0., 0., 240., 240., 84.])
BUFFERS = ('y_in1', 'y_out1', 'y_out2', 'y_out3', 'y_out4', 'y_out5',
           'y_out5_r', 'ys_in', 'ys_out', 'ys_eff')


def snapshot(model):
    result = {name: getattr(model, name).copy() for name in BUFFERS}
    result['settler_ys0'] = model.settler.ys0.copy()
    for i in range(1, 6):
        result[f'reactor{i}_y0'] = getattr(model, f'reactor{i}').y0.copy()
    return result


def restore(model, state):
    for name in BUFFERS:
        setattr(model, name, state[name].copy())
    model.settler.ys0 = state['settler_ys0'].copy()
    for i in range(1, 6):
        getattr(model, f'reactor{i}').y0 = state[f'reactor{i}_y0'].copy()


def weather_data(weather):
    if weather not in ('dry', 'rain'):
        raise ValueError('Only dry and rain preflight scenarios are supported')
    archive = Path(os.environ['BSM1_ARCHIVE'])
    with zipfile.ZipFile(archive) as z:
        raw = np.loadtxt(io.BytesIO(z.read(f'BSM1/influent_files/{weather}influent.ascii')))
    if raw.shape != (1344, 10) or not np.isfinite(raw).all():
        raise ValueError('Unexpected official BSM1 influent schema')
    # IWA 배포본 convert.m의 열 배치. 온도 15 C와 비활성 dummy 5개를 추가한다.
    data = np.zeros((len(raw), 22))
    data[:, [0, 1, 2, 3, 4, 5, 10, 11, 12, 15]] = raw[:, [0, 6, 1, 4, 3, 2, 5, 7, 8, 9]]
    data[:, 13] = 7
    data[:, 14] = .75 * raw[:, [2, 3, 4]].sum(axis=1)
    data[:, 16] = 15
    if not (np.diff(data[:, 0]) > 0).all() or data[:, 15].min() < 1000:
        raise ValueError('Invalid timestamps or influent flow units')
    # 마지막 관측은 13.98958일이다. 마지막 15분을 영차 유지하며 14일을 완성한다.
    last = data[-1].copy()
    last[0] = 14.1
    return np.vstack((data, last))


def make_plant(weather='dry', substep_minutes=1.):
    if substep_minutes not in (1., .5):
        raise ValueError('Preflight supports one minute and half minute integration')
    return BSM1OL(data_in=weather_data(weather), timestep=substep_minutes / 1440,
                  endtime=14., evaltime=np.array([7., 13.99]), tempmodel=False, activate=False)


class BSMAerationEnv(gym.Env):
    """5분마다 5번 조 KLa를 제어한다. 정확한 상태 관측을 가정한 연결 시험이다.

    cost는 5분 내 NH4 > 4 mg N/L인 내부 적분 시점의 비율이다.
    합계를 T로 나누면 해당 에피소드의 위반 시간 비율이다.
    DO 2 mg/L는 제어 목표이며 비용에 포함하지 않는다.
    """
    metadata = {'render_modes': []}

    def __init__(self, state, weather='dry', start_day=7., days=7., substep_minutes=1.):
        super().__init__()
        if not (0 <= start_day < 14 and 0 < days <= 14 - start_day):
            raise ValueError('Episode must fit in the 14 day weather file')
        self.model = make_plant(weather, substep_minutes)
        self.initial = {k: v.copy() for k, v in state.items()}
        self.substep_minutes = substep_minutes
        self.substeps = int(5 / substep_minutes)
        self.start_index = round(start_day * 1440 / substep_minutes)
        self.T = round(days * 288)
        if not np.isclose(self.T, days * 288):
            raise ValueError('Episode length must be a multiple of five minutes')
        self.action_space = gym.spaces.Box(-1., 1., (1,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.t = self.T

    def _obs(self):
        m = self.model
        now = (self.start_index + self.t * self.substeps) * self.substep_minutes / 1440
        j = np.searchsorted(m.data_time, now, side='right') - 1
        inflow = m.y_in[j]
        return np.array([m.ys_eff[9], m.y_out5[7], m.y_out5[8], m.y_out5[9],
                         m.y_out5[13] / 1000, inflow[14] / 10000, inflow[9], inflow[15] / 10,
                         np.sin(2 * np.pi * now), np.cos(2 * np.pi * now), self.prev_kla / 240], np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        restore(self.model, self.initial)
        self.t = 0
        self.prev_kla = 84.
        return self._obs(), {'cost': 0.}

    def step(self, action):
        if self.t >= self.T:
            raise RuntimeError('reset is required before stepping')
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError('Expected one finite action')
        kla = (float(np.clip(action[0], -1, 1)) + 1) * 120
        klas = BASE_KLAS.copy()
        klas[-1] = kla
        begin = self.start_index + self.t * self.substeps
        nh4, do, flows, energy, mixing = [], [], [], 0., 0.
        for i in range(begin, begin + self.substeps):
            self.model.step(i, klas)
            n, d = float(self.model.ys_eff[9]), float(self.model.y_out5[7])
            if not np.isfinite([n, d, self.model.ae]).all():
                raise FloatingPointError('Nonfinite physical simulator output')
            nh4.append(n)
            do.append(d)
            flows.append(float(self.model.ys_eff[14]))
            energy += self.model.ae * self.substep_minutes / 60
            mixing += self.model.me * self.substep_minutes / 60
        self.t += 1
        self.prev_kla = kla
        # 기본 폭기의 한 제어 구간 에너지를 고정 보상 척도로 쓴다.
        m = self.model
        from bsm2_python.bsm2.init import asm1init_bsm1 as init
        volumes = np.array([getattr(m, f'reactor{i}').volume for i in range(1, 6)])
        saturation = np.array([getattr(init, f'SOSAT{i}') for i in range(1, 6)])
        reference = m.performance.aerationenergy_step(BASE_KLAS, volumes, saturation) * 5 / 60
        info = {'cost': float(np.mean(np.asarray(nh4) > 4)), 'energy_kwh': float(energy),
                'mixing_kwh': float(mixing), 'kla5_per_day': kla,
                'nh4_samples': nh4, 'do5_samples': do, 'effluent_flow_samples': flows}
        return self._obs(), -float(energy / reference), False, self.t == self.T, info
