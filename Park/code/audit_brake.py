"""스텝 기록 평가와 차단 결정 감사 (Park 작성, Park/e44 ~ e46).

rollout_log: darrow_env.evaluate 와 같은 지표를 내면서 스텝마다 공기량, 다음 NH4, 다음 최대 DO, 위반을 기록한다.
    delay = d 이면 정책과 차단이 보는 관측 NH4 를 d 스텝 전 값으로 바꾼다 (NH4 분석계 지연, Park/e46). 첫 d 스텝은 첫 관측을 쓴다.
    환경 안의 실제 상태와 비용은 바꾸지 않는다.
AuditBrake: MinAirBrake 의 trim 경로 (대표 차단) 를 그대로 계산하면서 단계마다 공기량을 기록한다.
    단계: 정책 -> NH4 하한 (1 스텝 여유 검사) 과 회복 검사 -> DO 줄이기 -> 충돌 조정 -> 회복 계획.
    실행한 최종 공기량에서 차단 자신의 1 스텝 NH4 여유 검사와 DO 한도 검사를 다시 해 통과 여부를 남긴다 (검사와 실행의 일치, 팀 검토 4.3 절).
"""
import numpy as np

from darrow_env import COST_LIMIT, NH4_MAX, STATE, STEP_H, action_to_air, all_episodes
from min_brake import MinAirBrake, _Rows

STAGES = ('policy', 'nh4_floor', 'recover', 'do_trim', 'conflict', 'plan')


def rollout_log(env, days, act, shield=None, delay=0):
    obs = env.reset(all_episodes(days, env.dyn.n_members))
    N, T = len(env.episodes), env.T
    hist = [np.asarray(obs[:, 0]).copy()]
    L = {k: np.zeros((N, T)) for k in ('air', 'nh4', 'do', 'v_nh4', 'v_do', 'used')}
    for t in range(T):
        o = obs.copy()
        if delay:
            o[:, 0] = hist[max(t - delay, 0)]
        a = np.asarray(act(o), dtype=float).reshape(N)
        if shield is not None:
            a, used = shield(o, a, env)
            L['used'][:, t] = used
        obs, _, c, done, info = env.step(a)
        hist.append(np.asarray(obs[:, 0]).copy())
        L['air'][:, t], L['v_nh4'][:, t], L['v_do'][:, t] = info['air'], info['v_nh4'], info['v_do']
        L['nh4'][:, t], L['do'][:, t] = env.state[:, 0], env.state[:, 2:].max(axis=1)
    assert done
    cost = (L['v_nh4'] + L['v_do']).sum(axis=1)
    out = {'air_m3': float((L['air'].sum(axis=1) * STEP_H).mean()), 'cost': float(cost.mean()),
           'nh4_viol_pct': float((100 * L['v_nh4'].sum(axis=1) / T).mean()), 'do_viol_pct': float((100 * L['v_do'].sum(axis=1) / T).mean()),
           'days_within_limit_pct': float(100 * np.mean(cost <= COST_LIMIT)), 'n_episodes': N}
    if shield is not None:
        out['intervention_pct'] = 100 * float(L['used'].mean())
    L['day'] = np.array([str(d) for d, _ in env.episodes])
    L['member'] = np.array([m for _, m in env.episodes])
    return out, L


class AuditBrake(MinAirBrake):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert self.trim and not self.futile and self.bias is None and self.do_bias is None and not self.do_cap
        self.log = {k: [] for k in STAGES + ('nh4_ok', 'do_ok', 'r0', 'rr')}

    def required_air(self, obs, env):
        r0 = self._base_required_air(obs, env)
        rr = self._recover_air(obs, env)
        self._r0, self._rr = r0, rr
        return np.where(np.isnan(rr), r0, np.maximum(r0, rr))

    def trimmed_air(self, obs, env, policy_air):
        """MinAirBrake.trimmed_air (futile 없음) 와 같은 계산에 단계 기록을 더한다."""
        r = self.required_air(obs, env)
        air = np.maximum(policy_air, r)
        floor = air.copy()
        rows = np.flatnonzero(policy_air >= r)
        if len(rows):
            sub = _Rows(env, rows)
            _, do = self._rollout(obs[rows], sub, policy_air[rows, None], len(STATE))
            over = rows[do[:, 0] > self.do_limit]
            if len(over):
                _, do_grid = self._rollout(obs[over], _Rows(env, over), np.tile(self.grid, (len(over), 1)), len(STATE))
                inside = (self.grid[None, :] >= r[over, None]) & (self.grid[None, :] <= policy_air[over, None])
                ok = inside & (do_grid <= self.do_limit)
                best = np.where(ok, self.grid[None, :], -np.inf).max(axis=1)
                air[over] = np.where(np.isfinite(best), best, r[over])
        trim = air.copy()
        if self.conflict is not None:
            air = self._resolve_conflict(obs, env, policy_air, air)
        conf = air.copy()
        if self.plan is not None:
            air = self._plan_recovery(obs, env, air)
        rec = np.where(np.isnan(self._rr), -np.inf, self._rr)
        self.log['policy'].append(policy_air.copy())
        self.log['nh4_floor'].append(np.maximum(policy_air, self._r0))
        self.log['recover'].append(floor)
        self.log['do_trim'].append(trim)
        self.log['conflict'].append(conf)
        self.log['plan'].append(air.copy())
        self.log['r0'].append(self._r0.copy())
        self.log['rr'].append(rec)
        self.log['nh4_ok'].append(self.nh4_ok(obs, env, air))
        self.log['do_ok'].append(self._do_at(obs, env, air).max(axis=1) <= self.do_limit)
        return air

    def nh4_ok(self, obs, env, air):
        """공기량 air (N,) 에서 1 스텝 NH4 여유 검사 (다른 구성원 평균 + sigma 에 맞춘 여유 <= 4)."""
        t = env.t
        state = np.asarray(obs[:, :len(STATE)], dtype=float)
        x = np.column_stack([state, env.x['exo'][:, t], air[:, None] * env.x['frac'][:, t] / 1000.0, env.prev_zone / 1000.0])
        pred = np.stack([np.clip(state[:, 0] + self.dyn.models[m][0].predict(x), 0.0, self.dyn.hi[0]) for m in range(self.dyn.n_members)])
        keep = np.arange(self.dyn.n_members)[:, None] != env.member[None, :]
        n = keep.sum(axis=0)
        mu = np.where(keep, pred, 0.0).sum(axis=0) / n
        sigma = np.sqrt(np.where(keep, (pred - mu) ** 2, 0.0).sum(axis=0) / n)
        margin = np.full(len(mu), self.margins[-1][1])
        for upper, m in reversed(self.margins):
            margin = np.where(sigma < upper, m, margin)
        return mu + margin <= NH4_MAX

    def arrays(self):
        """기록을 (N, T) 배열로. 단계별 공기량, 마지막으로 공기량을 바꾼 단계 번호 (STAGES 순서), 검사 통과 여부."""
        A = {k: np.stack(v, axis=1) for k, v in self.log.items()}
        last = np.zeros(A['policy'].shape, dtype=int)
        for i in range(1, len(STAGES)):
            changed = np.abs(A[STAGES[i]] - A[STAGES[i - 1]]) > 1e-9
            last = np.where(changed, i, last)
        A['last'] = last
        return A


def final_air_check(brake_out, env_air):
    """AuditBrake 가 정한 최종 공기량과 환경이 실제로 받은 공기량의 최대 차이 (행동 변환 왕복 확인)."""
    return float(np.max(np.abs(brake_out - env_air)))


__all__ = ['rollout_log', 'AuditBrake', 'STAGES', 'final_air_check', 'action_to_air']
