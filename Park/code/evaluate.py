"""정책 평가와 지표 계산 (Park 작성).

평가 방식은 Lee/e01 과 같다. 공통 시작 상태에서 14일을 연속 운전하고 후반 7일
(스텝 2017 ~ 4032) 을 집계한다. 건기와 우기 두 날씨에서 잰다.

policy(obs) -> 행동. shield 를 주면 shield(obs, action) -> (행동, 개입 여부) 로 행동을 바꾼다.
"""
import numpy as np

from env import AerationEnv, NH4_LIMIT, STEPS_PER_DAY


def cvar(values, alpha=0.2):
    """상위 alpha 꼬리의 CVaR (큰 값이 나쁜 지표용).

    VaR + E[(x - VaR)+] / alpha 형태로, 경계에 걸친 표본의 질량을 분수로 반영한다.
    팀의 표(Yoo/e17, Lee/e01) 와 같은 정의다.
    """
    values = np.asarray(values, dtype=float)
    ordered = np.sort(values)
    index = max(0, int(np.ceil((1 - alpha) * len(values) - 1e-12)) - 1)
    var = ordered[index]
    return float(var + np.maximum(values - var, 0).mean() / alpha)


def metrics(infos, interventions=None):
    """스텝별 info 목록에서 지표를 계산한다."""
    nh4 = np.concatenate([x['nh4'] for x in infos])
    flow = np.concatenate([x['flow'] for x in infos])
    step_violation = np.array([x['violation'] for x in infos])
    steps = len(infos)
    days = steps / STEPS_PER_DAY
    result = {
        'days': days,
        'aeration_kwh_per_day': sum(x['energy_kwh'] for x in infos) / days,
        'mixing_kwh_per_day': sum(x['mixing_kwh'] for x in infos) / days,
        'flow_weighted_nh4': float(np.average(nh4, weights=flow)),
        'step_violation_percent': float(step_violation.mean() * 100),
        'minute_violation_percent': float((nh4 > NH4_LIMIT).mean() * 100),
        'violating_steps_per_day': float(step_violation.sum() / days),
        'kla5_mean': float(np.concatenate([x['kla5'] for x in infos]).mean()),
        'do5_mean': float(np.concatenate([x['do5'] for x in infos]).mean()),
    }
    if interventions is not None:
        result['intervention_percent'] = float(np.mean(interventions) * 100)
    return result


def run(initial, policy, weather='dry', controller=None, shield=None, days=14, score_from_day=7,
        archive=None):
    env = AerationEnv(initial, weather=weather, start_day=0, days=days, controller=controller,
                      archive=archive)
    obs, _ = env.reset()
    infos, interventions = [], []
    while env.t < env.T:
        action = np.asarray(policy(obs), dtype=float).reshape(1)
        used = False
        if shield is not None:
            action, used = shield(obs, action)
        obs, _, _, _, info = env.step(action)
        if env.t > score_from_day * STEPS_PER_DAY:
            infos.append(info)
            interventions.append(float(used))
    week = metrics(infos, interventions if shield is not None else None)
    daily = [metrics(infos[i:i + STEPS_PER_DAY]) for i in range(0, len(infos), STEPS_PER_DAY)]
    week['daily_nh4_cvar20'] = cvar([d['flow_weighted_nh4'] for d in daily])
    week['daily_violating_steps_cvar20'] = cvar([d['violating_steps_per_day'] for d in daily])
    return {'week': week, 'days': daily}


def evaluate(initial, policy, controller=None, shield=None, weathers=('dry', 'rain'), archive=None):
    return {w: run(initial, policy, w, controller, shield, archive=archive) for w in weathers}
