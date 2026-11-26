"""Aligned completed trade candles, preserving stale/missing observations."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/pas_strategy_audit_20261020'

def load(out=OUT):
    cfg=json.loads((out/'dataset.json').read_text());syms=cfg['symbols']
    times=np.arange(cfg['warmup_start_ms']+60000,cfg['cutoff_ms']+1,60000,dtype=np.int64)
    shape=(len(times),len(syms));d={k:np.full(shape,np.nan) for k in ('close','open','high','low')}
    d.update(times=times,volume=np.zeros(shape),active=np.zeros(shape,bool),age_ms=np.full(shape,1e20),present=np.zeros(shape,bool),funding_events=[])
    for j,s in enumerate(syms):
        rows=json.loads((out/'raw'/f'pas_candles_{s}.json').read_text())['data'];seen=set()
        for b in rows:
            if b['t'] in seen:raise ValueError('Duplicate candle')
            seen.add(b['t']);known=b['t']+60000
            if known>cfg['cutoff_ms'] or b['T']>=cfg['cutoff_ms']:continue
            i=int((known-times[0])//60000)
            if not 0<=i<len(times):continue
            assert times[i]==known
            for field,key in [('close','c'),('open','o'),('high','h'),('low','l')]:d[field][i,j]=float(b[key])
            d['volume'][i,j]=float(b['v']);d['active'][i,j]=b['n']>0 and float(b['v'])>0;d['present'][i,j]=True
        last_price=np.nan;last_trade=-1e20
        for i,t in enumerate(times):
            if d['active'][i,j]:last_trade=t
            if np.isfinite(d['close'][i,j]):last_price=d['close'][i,j]
            else:
                for key in ('close','open','high','low'):d[key][i,j]=last_price
            d['age_ms'][i,j]=t-last_trade
        for row in json.loads((out/'raw'/f'pas_funding_{s}.json').read_text())['data']:
            if cfg['start_ms']<row['time']<=cfg['cutoff_ms']:
                d['funding_events'].append({'time':int(row['time']),'j':j,'rate':float(row['fundingRate'])})
    d['funding_events'].sort(key=lambda x:(x['time'],x['j']))
    return cfg,d
