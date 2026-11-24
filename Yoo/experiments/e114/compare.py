"""Compare e113 winter episodes with similar savings on the same day and GBM member."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jio/fcsrl_src/results')
METRICS = ['saving', 'base_air', 'start_min', 'duration', 'pre', 'supply', 'nh15', 'nh60', 'nh180']
CALIPERS = {'primary': 1000, 'strict': 500, 'wide': 2000, 'base_close': 1000, 'do_ok': 1000}


def read(path):
    return json.loads(path.read_text())


def check_hash(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path


def longest(mask):
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return None
    pos = np.argmax(ends - starts)
    return int(starts[pos]), int(ends[pos])


def describe(air, baseline, nh, initial):
    span = longest(baseline - air >= 250)
    row = dict(start_min=np.nan, duration=np.nan, pre=np.nan, supply=np.nan,
               nh15=np.nan, nh60=np.nan, nh180=np.nan)
    if span is None:
        return row
    start, end = span
    row.update(start_min=start*15, duration=(end-start)*15, pre=nh[start-1] if start else initial,
               supply=air[start:end].mean())
    for steps in (1, 4, 12):
        if start + steps <= len(nh):
            row[f'nh{steps*15}'] = nh[start+steps-1]
    return row


def smoke():
    assert longest(np.array([0, 1, 1, 0, 1, 1])) == (1, 3)
    assert longest(np.zeros(4)) is None
    base = np.full(5, 3000.)
    nh = np.arange(1., 6.)
    row = describe(base-250, base, nh, .1)
    assert (row['duration'], row['pre'], row['nh15'], row['nh60']) == (75, .1, 1., 4.)
    row = describe(base-np.array([0, 0, 0, 0, 250]), base, nh, .1)
    assert row['pre'] == 4 and row['nh15'] == 5 and np.isnan(row['nh60'])


def load(source, prepared):
    plan = read(source / 'plan.json')
    dates = plan['winter']
    assert dates == sorted(set(dates)) and len(dates) == 40
    check_hash(prepared, plan['input_sha256']['data.pkl'])
    data = pd.read_pickle(prepared)
    assert list(data['days']['val']) == dates
    initial = {d: float(data['data'].loc[d, 'nh4'].iloc[0]) for d in dates}
    hashes = read(source / 'checksums.json')
    reference = pd.read_csv(HERE.parent / 'e113/per_seed.csv').query("block=='winter'").set_index(['seed','arm'])
    rows, paths = [], {}
    for seed in range(5):
        air, state, violations = {}, {}, {}
        for arm in ('uniform', 'weighted'):
            folder = source / arm / f'seed_{seed}'
            path = folder / 'winter_rollout.npz'
            check_hash(path, hashes[str(path.relative_to(source))])
            with np.load(path) as raw:
                air[arm], state[arm] = raw['air'], raw['states']
            assert air[arm].shape == (200, 95) and state[arm].shape == (200, 95, 5)
            assert np.isfinite(air[arm]).all() and np.isfinite(state[arm]).all()
            n = (state[arm][:, :, 0] > 4).sum(axis=1)
            d = (state[arm][:, :, 2:].max(axis=2) > 2).sum(axis=1)
            violations[arm] = np.column_stack([n, d])
            sums = air[arm].sum(axis=1) / 4
            ep = pd.DataFrame(read(folder / 'winter_episodes.json'))
            assert list(zip(ep.day, ep.member)) == [(day, m) for day in dates for m in range(5)]
            np.testing.assert_allclose(ep[['air_m3','nh4_cost','do_cost']], np.column_stack([sums,n,d]), atol=1e-8, rtol=0)
            np.testing.assert_allclose([sums.mean(), n.mean()*100/95, d.mean()*100/95],
                                       reference.loc[(seed,arm), ['air_m3','nh4_pct','do_pct']], atol=1e-8, rtol=0)
        for i in range(200):
            day, member = dates[i//5], i%5
            u, w = air['uniform'][i], air['weighted'][i]
            un, wn = (state[arm][i,:,0] for arm in ('uniform','weighted'))
            dn, dd = violations['weighted'][i] - violations['uniform'][i]
            rows.append(dict(day=day, member=member, seed=seed, saving=(u-w).sum()/4, base_air=u.sum()/4,
                             dn=dn, dd=dd, **describe(w,u,wn,initial[day])))
            paths[day,member,seed] = (u,w,np.r_[initial[day],un],np.r_[initial[day],wn])
    return pd.DataFrame(rows), paths, dates


def make_pairs(episodes):
    pairs = []
    for (day, member), group in episodes.groupby(['day','member']):
        for a, b in combinations(group.itertuples(), 2):
            if min(a.saving,b.saving) <= 0:
                continue
            gap = abs(a.saving-b.saving)
            if gap > .25*(a.saving+b.saving)/2 or gap > max(CALIPERS.values()):
                continue
            good, bad = sorted([a,b], key=lambda r: (r.dn,r.seed))
            opposed = good.dn < 0 < bad.dn
            row = dict(day=day, member=member, seed_lo=good.seed, seed_hi=bad.seed, gap=gap,
                       opposite=opposed, nh_lo=good.dn, nh_hi=bad.dn, do_lo=good.dd, do_hi=bad.dd)
            for key in METRICS:
                row['lo_'+key], row['hi_'+key] = getattr(good,key), getattr(bad,key)
            pairs.append(row)
    return pd.DataFrame(pairs)


def summaries(pairs, dates):
    summaries, counts, signals = [], [], []
    for name, width in CALIPERS.items():
        selected = pairs[pairs.gap <= width]
        if name == 'base_close':
            selected = selected[(selected.hi_base_air-selected.lo_base_air).abs() <= 2500]
        for period, days in [('all', dates),('early', dates[:20]),('late', dates[20:])]:
            candidates = selected[selected.day.isin(days)]
            g = candidates[candidates.opposite]
            if name == 'do_ok':
                g = g[g.do_lo <= 0]
            counts.append(dict(rule=name,period=period,candidates=len(candidates),discordant=len(g),days=g.day.nunique(),
                               pairs01_14=int(((g.seed_hi==1)&g.seed_lo.isin([0,4])).sum())))
            for key in METRICS:
                cols = ['lo_'+key, 'hi_'+key]
                valid = g.dropna(subset=cols)
                daily = valid.groupby(['day','member'])[cols].mean().groupby('day').mean()
                diff = daily[cols[1]] - daily[cols[0]]
                n_combo = len({tuple(sorted(p)) for p in zip(valid.seed_hi, valid.seed_lo)})
                summaries.append(dict(rule=name,period=period,metric=key,improved=daily[cols[0]].mean(),
                                      worsened=daily[cols[1]].mean(),diff=diff.mean(),pairs=len(valid),days=len(daily),
                                      positive_days=int((diff>0).sum()),seed_combinations=n_combo))
    summary = pd.DataFrame(summaries)
    for key in ('pre','duration'):
        g = summary.query("rule=='primary' and period!='all' and metric==@key")
        passed = ((g.days>=5)&(g.seed_combinations>=3)&(g['diff']>0)&(g.positive_days>g.days/2))
        signals.append(dict(metric=key, both_halves=bool(passed.all() and len(g)==2)))
    return summary, pd.DataFrame(counts), signals


def run(source, prepared, out):
    smoke()
    episodes, paths, dates = load(source, prepared)
    pairs = make_pairs(episodes)
    summary, counts, signals = summaries(pairs, dates)
    out.mkdir(parents=True, exist_ok=False)
    episodes.to_csv(out/'episodes.csv', index=False)
    pairs.to_csv(out/'pairs.csv', index=False)
    summary.to_csv(out/'summary.csv', index=False)
    counts.to_csv(out/'counts.csv', index=False)
    (out/'signals.json').write_text(json.dumps(signals, indent=2)+'\n')
    primary = pairs[pairs.opposite & (pairs.gap<=1000)].sort_values(['gap','day','member','seed_lo','seed_hi'])
    parts = []
    for case in primary.itertuples():
        for side in ('lo', 'hi'):
            seed = getattr(case, 'seed_'+side)
            u, w, un, wn = paths[case.day,case.member,seed]
            span = longest(u-w >= 250)
            if span is None:
                continue
            first, end = span
            cost = (wn[1:]>4).astype(int) - (un[1:]>4).astype(int)
            before, during, after = (int(part.sum()) for part in (cost[:first],cost[first:end],cost[end:]))
            assert before+during+after == getattr(case,'nh_'+side)
            parts.append(dict(day=case.day,member=case.member,seed=seed,other=getattr(case,'seed_'+('hi' if side=='lo' else 'lo')),
                              side=side,before=before,during=during,after=after))
    pd.DataFrame(parts).to_csv(out/'phases.csv', index=False)
    if not primary.empty:
        example = primary.iloc[0]
        rows = []
        for seed in (example.seed_lo, example.seed_hi):
            u,w,un,wn = paths[example.day,example.member,seed]
            for t in range(95):
                rows.append(dict(day=example.day, member=example.member, seed=seed, step=t, uniform_air=u[t],
                                 weighted_air=w[t], uniform_pre=un[t], weighted_pre=wn[t],uniform_next=un[t+1],
                                 weighted_next=wn[t+1]))
        pd.DataFrame(rows).to_csv(out/'example.csv', index=False)
    print(counts.to_string(index=False))
    print(summary.query("rule=='primary' and metric in ['pre','duration','supply','nh60','base_air']").round(3).to_string(index=False))
    print(signals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT/'e113/main')
    parser.add_argument('--prepared', type=Path, default=ROOT/'e50/prepared/data.pkl')
    args = parser.parse_args()
    run(args.source, args.prepared, args.out)
