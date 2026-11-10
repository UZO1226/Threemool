"""Kaggle 대회 "WWTP model: mechanistic, data-driven, or hybrid?" (DARROW, Tilburg) 온라인 자료를 정리한다.

입력: train_val_test_online_dataset.csv (탭 구분, 15분 간격, 2021-05-05 에서 2022-01-30, 결측과 공백 없음)
출력: 학습, 검증, 평가 날짜로 나눈 하루(96스텝) 표. evaluation_*.csv 는 정답(DO, NH4, NO3)이 없는 대회 제출용이라 쓰지 않는다.

분할은 날짜 순서대로 70%, 15%, 15% 다 (미래 정보가 학습에 섞이지 않게 한다).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STEPS_PER_DAY = 96
AIR_COLS = [f'Q_air_{i}' for i in range(1, 6)]
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def load(csv):
    d = pd.read_csv(csv, sep='\t', parse_dates=['DATETIME']).set_index('DATETIME').sort_index()
    if d.isna().any().any():
        raise ValueError('Unexpected missing values')
    step = d.index.to_series().diff().dropna()
    if not (step == pd.Timedelta('15min')).all():
        raise ValueError('Expected a regular 15 minute grid')
    out = pd.DataFrame({'nh4': d.NH4, 'no3': d.NO3, 'do1': d.DO_1, 'do2': d.DO_2, 'do3': d.DO_3,
                        'q_in': d.Q_inf, 'temp': d.Temp, 'air': d[AIR_COLS].sum(axis=1)})
    for i in range(1, 6):
        out[f'air{i}'] = d[f'Q_air_{i}']
    h = out.index.hour + out.index.minute / 60
    out['sin'], out['cos'] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    return out


INFLUENT_COLS = ['S_NH', 'S_ND', 'X_ND', 'S_S', 'X_S']


def add_influent(table, csv):
    """dynamic_influent 파일(ASM1 유입 성분, 14.4분 간격)을 온라인 자료의 15분 격자에 선형 보간해 붙인다.
    유입 유량 Q_in(m3/d)과 온라인 Q_inf(m3/h)의 상관이 시차 0 에서 0.998 로 시간이 정확히 맞는 것을 확인했다."""
    inf = pd.read_csv(csv, sep='	', index_col=0, low_memory=False).iloc[1:].apply(pd.to_numeric, errors='coerce')
    inf = inf.reset_index(drop=True)
    t = inf['Time'].to_numpy(float)
    grid = np.arange(len(table)) * 15 / 1440
    check = float(np.corrcoef(np.interp(grid, t, inf['Q_in'].to_numpy(float)), table.q_in.to_numpy(float))[0, 1])
    if check < 0.99:
        raise ValueError(f'Influent file does not align with online data (corr {check:.3f})')
    for c in INFLUENT_COLS:
        table['inf_' + c.lower()] = np.interp(grid, t, inf[c].to_numpy(float))
    return table, check


def complete_days(table):
    days = []
    for day, g in table.groupby(table.index.date):
        if len(g) == STEPS_PER_DAY:
            days.append(str(day))
    return days


def split_days(days):
    n = len(days)
    a = int(round(n * SPLIT_FRACTIONS[0]))
    b = a + int(round(n * SPLIT_FRACTIONS[1]))
    return {'train': days[:a], 'val': days[a:b], 'test': days[b:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--influent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    table = load(a.csv)
    table, align = add_influent(table, a.influent)
    print('유입 파일 시간 정렬 상관: %.4f' % align)
    days = complete_days(table)
    splits = split_days(days)
    zone = [f'air{i}' for i in range(1, 6)]
    train_table = pd.concat([table.loc[d] for d in splits['train']])
    fractions = (train_table[zone].median() / train_table[zone].median().sum()).to_dict()
    pd.to_pickle({'data': table, 'days': splits, 'zone_fractions': fractions,
                  'steps_per_day': STEPS_PER_DAY}, a.out)
    for name, ds in splits.items():
        print(f'{name}: 완전한 날 {len(ds)}일 ({ds[0]} ~ {ds[-1]})')
    print('공기량 구간별 비율(중앙값 기준):', {k: round(v, 3) for k, v in fractions.items()})
    print('총 공기량 분위수 50/99/max:', [round(float(table.air.quantile(q)), 0) for q in (.5, .99, 1.0)])


if __name__ == '__main__':
    main()
