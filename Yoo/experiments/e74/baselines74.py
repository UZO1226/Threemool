"""e74 비교군: e68 비교군 코드 (기록 운전, 고정 공기, ABAC 격자) 를 새 비용 (env74) 으로 다시 고른다.

규칙은 e68 과 같고 한도만 바뀐다: dev 에서 하루 비용 스텝 (NH4 > 4 또는 예산 초과) <= 4.75 인 조합 중 공기가 가장 적은 것.
격자 끝 검사도 e68 과 같다. e68 에서 고른 ABAC (설정값 2.9) 도 새 비용으로 다시 평가해 함께 남긴다. holdout, test 는 읽지 않는다.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'e68'))
import env74  # noqa: E402
import baselines as Bs  # noqa: E402

E68_ABAC = dict(sp=2.9, kp=1000.0, ki=12800.0)


def select(rows):
    ok = [r for r in rows if r[3]['step_cost'] <= env74.EVAL_LIMIT]
    return min(ok, key=lambda r: r[3]['air_m3']) if ok else None


def main(a):
    assert Bs.E.rollout is env74.rollout74 and Bs.E.DailyEnv is env74.DailyEnv74
    a.out.mkdir(parents=True, exist_ok=False)
    jobs = [('abac', dict(sp=sp, kp=kp, ki=ki), 'dev') for sp, kp, ki in itertools.product(Bs.SETPOINTS, Bs.KP, Bs.KI)]
    jobs += [('constant', dict(air=float(x)), 'dev') for x in Bs.CONSTANT]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(Bs._eval, jobs))
    (a.out / 'grid.json').write_text(json.dumps([dict(kind=k, params=p, dev=r) for k, p, _, r in rows], indent=0))
    chosen = {}
    for kind in ('abac', 'constant'):
        best = select([r for r in rows if r[0] == kind])
        assert best is not None, f'{kind}: 한도를 만족하는 조합이 없다'
        chosen[kind] = best[1]
    p = chosen['abac']
    edge = [n for n, v, grid in (('sp', p['sp'], Bs.SETPOINTS), ('kp', p['kp'], Bs.KP), ('ki', p['ki'], Bs.KI))
            if (v == max(grid) and n != 'ki') or (v == min(grid) and n != 'kp')]
    assert not edge, f'ABAC 선택값이 격자 끝이다: {edge} {p}'
    assert Bs.CONSTANT[0] < chosen['constant']['air'] < Bs.CONSTANT[-1]
    chosen['record'] = {}
    chosen['abac_e68'] = E68_ABAC
    final = {k: dict(params=p) for k, p in chosen.items()}
    jobs = [(('abac' if k == 'abac_e68' else k), p, s, k) for k, p in chosen.items() for s in ('dev', 'val')]
    with ProcessPoolExecutor(a.jobs) as ex:
        for (kind, params, split, res), (_, _, _, key) in zip(ex.map(Bs._eval, [j[:3] for j in jobs]), jobs):
            final[key][split] = res
    (a.out / 'baselines.json').write_text(json.dumps(dict(limit=env74.EVAL_LIMIT, chosen=final, holdout_evaluated=False, test_evaluated=False), indent=1))
    for k, v in final.items():
        print(k, v['params'], {s: (round(v[s]['air_m3']), round(v[s]['step_cost'], 3), round(v[s]['nh4_excess_mg_h_L'], 2)) for s in ('dev', 'val')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=14)
    main(p.parse_args())
