"""Lee 의 변경 한 가지: 안전 비용 정의를 평균 농도에서 순간 위반 지시값으로 바꾼다.

기준(mean): Yoo/e17 의 비용. 한 스텝 비용은 정규화한 NH4 부하이고 하루 합계는
유량 가중 평균 NH4 / 4 x 288 이다. 하루 한도 288 은 평균 NH4 <= 4 mg N/L 를 뜻한다.
비용이 연속값이라 FCSRL 의 feasibility 해석(명제 4.1)이 성립하지 않는다.

변경(binary): 한 스텝 비용은 그 5분 구간에서 NH4 > 4 mg N/L 인 내부 적분 시점이
하나라도 있으면 1, 없으면 0 이다. 하루 합계는 위반한 제어 스텝의 개수이고
한도는 하루 288 스텝의 25% 인 72 스텝이다. 비용이 0 또는 1 이므로 FCSRL 의
feasibility 점수 F 는 "앞으로 한 번이라도 위반할 확률"이 되고 1 - F 는 계속
안전할 확률이 된다 (FCSRL 논문 명제 4.1 의 전제).

두 모드는 보상(폭기 에너지의 음수), 관측, 행동, 환경, 하루 에피소드 경계, 학습
예산, 신경망 구조를 모두 공유한다. 바뀌는 것은 비용 정의와 그에 따른 한도,
그리고 같은 상대 제약 오차에 같은 PID 응답을 주기 위한 단위 환산뿐이다.

환경과 평가 지표는 Yoo 의 MeanEnv 와 episode_metrics 를 그대로 호출한다
(Lee/code/bsm/README.md 의 출처 표 참고). 여기서는 정보 항목만 덧붙인다.
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fcsrl_compat  # noqa: F401  fcsrl 보다 먼저 가져와야 한다
from bsm_mean import MeanEnv, episode_metrics, normalized_cost  # noqa: E402

# bsm_mean 은 가져올 때 Yoo/code/wwtp 를 sys.path 맨 앞에 넣는다. 이 실험은 Lee 폴더의
# 복사본(내용은 같고 verify.py 가 sha256 일치를 검사한다)을 쓰므로 순서를 되돌린다.
while str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

NH4_LIMIT = 4.0
# DO 상한. BSM1 의 표준 DO 목표는 2.0 mg/L 이고, 목표 2.0 의 DO PI 기준 제어는 2.5 를
# 한 번도 넘지 않는다(건기 평가 0.0%). 반면 최대 폭기는 88.5% 넘는다. 즉 잘 조정된
# 제어는 벌하지 않고 과폭기만 제약하는 값이다. 측정값은 Lee/experiments/e04/README.md.
DO_LIMIT = 2.5
STEPS_PER_DAY = 288
# 사전 고정값. 근거는 Lee/experiments/e01/README.md 의 기준 제어 측정표다.
BINARY_DAILY_LIMIT = 72.0
MEAN_DAILY_LIMIT = 288.0
# Yoo 의 PID 계수는 한도 15 를 기준으로 조정돼 있다. 같은 상대 오차에 같은 응답을
# 주도록 한도 비율로 환산한다 (Yoo/e17 과 같은 규칙).
BASE_PID = (0.02, 0.005, 0.01)
BASE_PID_LIMIT = 15.0

MODES = ('mean', 'binary', 'binary_do')
BINARY_MODES = ('binary', 'binary_do')


class BinaryEnv(MeanEnv):
    """MeanEnv 와 물리, 보상, 관측, 에피소드 경계가 같다. 위반 지시값만 덧붙인다."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        nh4 = info['effluent'][:, 9]
        info['violation'] = float((nh4 > NH4_LIMIT).any())
        info['violation_do'] = float((info['do5'] > DO_LIMIT).any())
        info['violation_joint'] = float(info['violation'] or info['violation_do'])
        info['nh4_max'] = float(nh4.max())
        info['do5_max'] = float(info['do5'].max())
        return obs, reward, terminated, truncated, info


def daily_limit(mode):
    check_mode(mode)
    return BINARY_DAILY_LIMIT if mode in BINARY_MODES else MEAN_DAILY_LIMIT


def pid_gains(mode, limit=None):
    """한도를 지정하지 않으면 모드의 기본 한도를 쓴다 (e01, e02 와 같은 값)."""
    limit = daily_limit(mode) if limit is None else float(limit)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('Daily cost limit must be a positive finite number')
    scale = BASE_PID_LIMIT / limit
    return {name: value * scale for name, value in zip(('pid_kp', 'pid_ki', 'pid_kd'), BASE_PID)}


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown cost mode {mode!r}; expected one of {MODES}')


def step_cost(mode, info):
    """수집 중 한 스텝에 기록하는 값. mean 모드는 하루 끝에 유량으로 정규화한다."""
    check_mode(mode)
    value = float({'mean': info.get('load'), 'binary': info.get('violation'),
                   'binary_do': info.get('violation_joint')}[mode])
    if not np.isfinite(value) or value < 0:
        raise ValueError('Step cost must be finite and nonnegative')
    if mode in BINARY_MODES and value not in (0.0, 1.0):
        raise ValueError('Binary cost must be exactly 0 or 1')
    return value


def finalize_day(mode, cost_block, flow_block):
    """완결된 하루 블록 (288, n_env) 의 비용을 확정한다."""
    check_mode(mode)
    cost_block, flow_block = np.asarray(cost_block), np.asarray(flow_block)
    if cost_block.shape != flow_block.shape or len(cost_block) != STEPS_PER_DAY:
        raise ValueError('Expected one full day of matching cost and flow blocks')
    if mode == 'mean':
        return normalized_cost(cost_block, flow_block)
    if not np.isin(cost_block, (0.0, 1.0)).all():
        raise ValueError('Binary cost block must contain only 0 and 1')
    return cost_block


def violation_metrics(infos):
    """Yoo 의 episode_metrics 에 제어 스텝 단위 위반 지표를 더한다."""
    base = episode_metrics(infos)
    steps = np.array([float((x['effluent'][:, 9] > NH4_LIMIT).any()) for x in infos])
    do_steps = np.array([float((x['do5'] > DO_LIMIT).any()) for x in infos])
    joint = np.maximum(steps, do_steps)
    days = len(steps) / STEPS_PER_DAY
    daily = steps.reshape(-1, STEPS_PER_DAY).sum(axis=1) if float(days).is_integer() else None
    base['step_violation_percent'] = float(steps.mean() * 100)
    base['violating_steps_per_day'] = float(steps.sum() / days)
    base['do_step_violation_percent'] = float(do_steps.mean() * 100)
    base['do_violating_steps_per_day'] = float(do_steps.sum() / days)
    base['joint_step_violation_percent'] = float(joint.mean() * 100)
    base['joint_violating_steps_per_day'] = float(joint.sum() / days)
    base['do_limit'] = DO_LIMIT
    if daily is not None:
        base['daily_violating_steps'] = daily.tolist()
        base['days_within_binary_limit_percent'] = float((daily <= BINARY_DAILY_LIMIT).mean() * 100)
        daily_joint = joint.reshape(-1, STEPS_PER_DAY).sum(axis=1)
        base['daily_joint_violating_steps'] = daily_joint.tolist()
    return base


def evaluate(initial, policy, controller=False, weathers=('dry', 'rain'), days=14., score_from_day=7):
    """두 날씨에서 연속 운전하고 후반 구간을 집계한다. Yoo/e17 evaluate 와 같은 구성이다."""
    from ppo_tail import empirical_cvar

    result = {}
    for weather in weathers:
        env = BinaryEnv(initial, weather=weather, days=days, controller=controller)
        obs, _ = env.reset()
        infos = []
        for _ in range(env.T):
            obs, _, _, _, info = env.step(np.asarray(policy(obs)).reshape(1))
            if env.t > score_from_day * STEPS_PER_DAY:
                infos.append(info)
        week = violation_metrics(infos)
        daily = [violation_metrics(infos[i:i + STEPS_PER_DAY])
                 for i in range(0, len(infos), STEPS_PER_DAY)]
        week['daily_nh4_cvar20'] = empirical_cvar([d['flow_weighted_nh4'] for d in daily], .2)[1]
        week['daily_violating_steps_cvar20'] = empirical_cvar(
            [d['violating_steps_per_day'] for d in daily], .2)[1]
        week['daily_joint_violating_steps_cvar20'] = empirical_cvar(
            [d['joint_violating_steps_per_day'] for d in daily], .2)[1]
        result[weather] = {'week': week, 'days': daily}
    return result


def satisfied(mode, week, limit=None):
    """해당 한도를 평가 구간 평균 기준으로 만족했는가. 기본값은 모드의 기본 한도다."""
    check_mode(mode)
    limit = daily_limit(mode) if limit is None else float(limit)
    if mode == 'binary':
        return bool(week['violating_steps_per_day'] <= limit)
    if mode == 'binary_do':
        return bool(week['joint_violating_steps_per_day'] <= limit)
    return bool(week['cost_ratio'] * STEPS_PER_DAY <= limit)
