"""기존 전처리에서 제외된 유입 및 유출 경로와 SS 열의 사용 가능성을 점검한다."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import pandas as pd
from history_trial import write

COLUMNS={'inlet':'BIOLOGY.LINE 3.PROCESSPHASE.INLET TANK',
         'outlet':'BIOLOGY.LINE 3.PROCESSPHASE.OUTLET TANK',
         'ss':'BIOLOGY.LINE 3 TANK 1.SS'}


def main(a):
    if a.out.exists():raise ValueError(f'Refusing overwrite {a.out}')
    data=pd.read_pickle(a.data)
    cols=['time']+[v+s for v in COLUMNS.values() for s in (' value',' quality')]
    raw=pd.read_csv(a.raw,usecols=cols)
    raw.index=pd.to_datetime(raw.pop('time'),utc=True).dt.tz_convert('Europe/Copenhagen')
    raw=raw.loc['2022-09-15':'2023-12-31']
    features={}
    for name,col in COLUMNS.items():
        value=raw[col+' value'].where(raw[col+' quality'].eq(0))
        if name=='ss':features[name]=value.resample('5min').mean()
        else:
            for code in (0,1,2):
                features[f'{name}_{code}']=(value==code).astype(float).where(value.notna()).resample('5min').mean()
    aux=pd.DataFrame(features)
    days=data['days']['train'];partition={'fit':days[:263],'inner_holdout':days[264:]}
    report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'raw_sha256':hashlib.sha256(a.raw.read_bytes()).hexdigest(),
            'processed_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
            'outer_val_used':False,'test_used':False,'imputation':'none',
            'quality_policy':'keep quality == 0 only','coverage':{},'fit_phase_inlet_groups':[]}
    for split,split_days in partition.items():
        indexes=data['data'].loc[data['data'].index.strftime('%Y-%m-%d').isin(split_days)].index
        g=aux.reindex(indexes)
        report['coverage'][split]={'days':len(split_days),'rows':len(g),
            'valid_fraction':{name:float(g[name].notna().mean()) for name in g},
            'complete_days_all_features':int(g.notna().all(1).groupby(g.index.date).all().sum())}
    pairs=[]
    for day in partition['fit']:
        actual=data['data'].loc[day]
        part=actual.iloc[:-1].copy()
        part['delta_nh4']=actual.nh4.to_numpy()[1:]-actual.nh4.to_numpy()[:-1]
        part=part.join(aux)
        pairs.append(part)
    frame=pd.concat(pairs)
    for phase in (0,1,2):
        for tank in (1,2):
            g=frame.loc[(frame.phase==phase)&(frame[f'inlet_{tank}']>=.8)]
            if len(g)==0:continue
            report['fit_phase_inlet_groups'].append({'phase':phase,'inlet_tank':tank,'n':len(g),
                 'mean_delta_nh4':float(g.delta_nh4.mean()),'median_delta_nh4':float(g.delta_nh4.median()),
                 'nh4_rising_fraction':float((g.delta_nh4>0).mean()),
                 'mean_current_nh4':float(g.nh4.mean()),'mean_do':float(g.do.mean()),
                 'mean_air':float(g.air.mean()),'mean_flow':float(g.q_in.mean()),'mean_temp':float(g.temp.mean())})
    report['completed_at']=datetime.now(timezone.utc).isoformat()
    write(a.out,report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('raw','data','out'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
