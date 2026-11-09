"""BSM1 폭기 제어 환경 (Park 작성).

5분마다 5번 조의 산소 전달 계수 KLa5 (0 ~ 240 /day) 를 정한다. 공정 계산은
`bsm2-python` 의 BSM1OL 이 1분 간격으로 한다. 환경 정의(관측 11개, 보상, 이진 비용,
하루 에피소드)는 팀 결과와 비교할 수 있도록 Yoo/e16, Yoo/e17, Lee/e01 과 같게 맞췄다
(`SOURCE.md` 참고). 코드는 복사하지 않고 새로 작성했다.

- 행동: [-1, 1] 을 KLa5 = (a + 1) x 120 으로 바꾼다. 1 ~ 4번 조 KLa 는 고정 (0, 0, 240, 240)
- 보상: 그 5분의 폭기 에너지를 기본 폭기(KLa5 = 84) 5분 에너지로 나눈 값의 음수
- 비용: 그 5분 안의 1분 표본 중 방류 NH4 > 4 mg N/L 가 하나라도 있으면 1, 없으면 0
- 에피소드: `days` 일 x 288 스텝. 끝나면 terminated = True (다음 날로 가치를 잇지 않는다)
"""
import io
import os
from pathlib import Path
import zipfile

import gymnasium as gym
import numpy as np

NH4_LIMIT = 4.0               # mg N/L
STEPS_PER_DAY = 288           # 5분 x 288 = 하루
MINUTES_PER_STEP = 5
KLA_MAX = 240.0
FIXED_KLAS = np.array([0., 0., 240., 240., 84.])   # 기본 폭기. 5번째 값만 행동으로 바꾼다

# 공정 상태 저장과 복원에 쓰는 BSM1OL 내부 버퍼 이름
_BUFFERS = ('y_in1', 'y_out1', 'y_out2', 'y_out3', 'y_out4', 'y_out5',
            'y_out5_r', 'ys_in', 'ys_out', 'ys_eff')


def kla_to_action(kla):
    return kla / (KLA_MAX / 2) - 1


def action_to_kla(action):
    return (float(np.clip(action, -1, 1)) + 1) * (KLA_MAX / 2)


def load_influent(weather, archive=None):
    """IWA 배포본 BSM1.zip 의 유입 자료를 BSM1OL 입력 형식(22열)으로 바꾼다.

    열 배치는 배포본의 convert.m 을 따른다. 수온은 15 C 고정이다.
    """
    if weather not in ('dry', 'rain'):
        raise ValueError(f'weather 는 dry 또는 rain 이어야 한다: {weather!r}')
    archive = Path(archive or os.environ['BSM1_ARCHIVE'])
    with zipfile.ZipFile(archive) as z:
        raw = np.loadtxt(io.BytesIO(z.read(f'BSM1/influent_files/{weather}influent.ascii')))
    if raw.shape != (1344, 10) or not np.isfinite(raw).all():
        raise ValueError('BSM1 유입 자료의 형식이 예상과 다르다')
    data = np.zeros((len(raw), 22))
    data[:, [0, 1, 2, 3, 4, 5, 10, 11, 12, 15]] = raw[:, [0, 6, 1, 4, 3, 2, 5, 7, 8, 9]]
    data[:, 13] = 7                                   # 알칼리도
    data[:, 14] = 0.75 * raw[:, [2, 3, 4]].sum(axis=1)  # TSS
    data[:, 16] = 15                                  # 수온
    # 마지막 기록은 13.99일 근처다. 14일을 채우도록 마지막 값을 그대로 늘린다.
    last = data[-1].copy()
    last[0] = 14.1
    return np.vstack((data, last))


def make_plant(weather='dry', archive=None):
    from bsm2_python import BSM1OL
    return BSM1OL(data_in=load_influent(weather, archive), timestep=1 / 1440, endtime=14.,
                  evaltime=np.array([7., 13.99]), tempmodel=False, activate=False)


def snapshot(model):
    """공정 상태 전체를 dict 로 복사한다 (np.savez 로 저장할 수 있다)."""
    state = {name: getattr(model, name).copy() for name in _BUFFERS}
    state['settler_ys0'] = model.settler.ys0.copy()
    for i in range(1, 6):
        state[f'reactor{i}_y0'] = getattr(model, f'reactor{i}').y0.copy()
    return state


def restore(model, state):
    for name in _BUFFERS:
        setattr(model, name, np.array(state[name]).copy())
    model.settler.ys0 = np.array(state['settler_ys0']).copy()
    for i in range(1, 6):
        getattr(model, f'reactor{i}').y0 = np.array(state[f'reactor{i}_y0']).copy()


def reference_energy_per_step(model):
    """기본 폭기(FIXED_KLAS) 로 5분 동안 쓰는 폭기 에너지 kWh. 보상의 척도다."""
    from bsm2_python.bsm2.init import asm1init_bsm1 as init
    volumes = np.array([getattr(model, f'reactor{i}').volume for i in range(1, 6)])
    saturation = np.array([getattr(init, f'SOSAT{i}') for i in range(1, 6)])
    return model.performance.aerationenergy_step(FIXED_KLAS, volumes, saturation) * MINUTES_PER_STEP / 60


class AerationEnv(gym.Env):
    """BSM1 5번 조 폭기 제어.

    state: snapshot() 으로 만든 시작 상태. reset() 마다 이 상태로 돌아간다.
    start_day: 유입 자료에서 몇 번째 날부터 시작할지 (0 ~ 13).
    controller: 주면 정책 행동을 무시하고 1분마다 controller(DO) 가 KLa5 를 정한다 (기준 제어용).
    """
    metadata = {'render_modes': []}

    def __init__(self, state, weather='dry', start_day=0, days=1, controller=None, archive=None):
        super().__init__()
        if not (0 <= start_day < 14 and 0 < days <= 14 - start_day):
            raise ValueError('에피소드가 14일 유입 자료 안에 들어가야 한다')
        self.model = make_plant(weather, archive)
        self.initial = {k: np.array(v).copy() for k, v in state.items()}
        self.start_day = start_day
        self.T = int(round(days * STEPS_PER_DAY))
        self.controller_factory = controller
        self.controller = None
        self.energy_reference = reference_energy_per_step(self.model)
        self.action_space = gym.spaces.Box(-1., 1., (1,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.t = self.T  # reset 전에는 step 할 수 없다

    def set_start(self, state, start_day):
        """다음 reset 의 시작 상태와 날짜를 바꾼다 (학습 중 하루씩 날을 바꿀 때 쓴다)."""
        self.initial = {k: np.array(v).copy() for k, v in state.items()}
        self.start_day = start_day

    def _minute(self):
        return self.start_day * 1440 + self.t * MINUTES_PER_STEP

    def _obs(self):
        m = self.model
        now = self._minute() / 1440
        j = np.searchsorted(m.data_time, now, side='right') - 1
        inflow = m.y_in[j]
        return np.array([
            m.ys_eff[9],              # 방류 NH4
            m.y_out5[7],              # 5번 조 DO
            m.y_out5[8],              # 5번 조 질산성 질소
            m.y_out5[9],              # 5번 조 NH4
            m.y_out5[13] / 1000,      # 5번 조 부유물
            inflow[14] / 10000,       # 유입 부유물
            inflow[9],                # 유입 NH4
            inflow[15] / 10,          # 유입 유량
            np.sin(2 * np.pi * now),  # 하루 중 시각
            np.cos(2 * np.pi * now),
            self.prev_kla / KLA_MAX,  # 직전 KLa5
        ], np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        restore(self.model, self.initial)
        self.t = 0
        self.prev_kla = FIXED_KLAS[-1]
        self.controller = self.controller_factory() if self.controller_factory else None
        return self._obs(), {}

    def step(self, action):
        if self.t >= self.T:
            raise RuntimeError('에피소드가 끝났다. reset() 을 먼저 호출한다')
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError('행동은 유한한 값 하나여야 한다')
        kla = action_to_kla(action[0])
        begin = self._minute()
        effluent, do5, klas5 = [], [], []
        energy = mixing = 0.
        for minute in range(begin, begin + MINUTES_PER_STEP):
            if self.controller is not None:
                kla = self.controller(float(self.model.y_out5[7]))
            klas = FIXED_KLAS.copy()
            klas[-1] = kla
            self.model.step(minute, klas)
            effluent.append(self.model.ys_eff.copy())
            do5.append(float(self.model.y_out5[7]))
            klas5.append(kla)
            energy += self.model.ae / 60
            mixing += self.model.me / 60
        effluent = np.asarray(effluent)
        nh4, flow = effluent[:, 9], effluent[:, 14]
        if not (np.isfinite(effluent).all() and np.isfinite([energy, *do5]).all()):
            raise FloatingPointError('BSM1 계산 결과에 유한하지 않은 값이 있다')
        self.t += 1
        self.prev_kla = kla
        violation = float((nh4 > NH4_LIMIT).any())
        info = {'cost': violation, 'violation': violation, 'energy_kwh': float(energy),
                'mixing_kwh': float(mixing), 'nh4': nh4, 'flow': flow,
                'do5': np.array(do5), 'kla5': np.array(klas5)}
        reward = -energy / self.energy_reference
        return self._obs(), float(reward), self.t == self.T, False, info
