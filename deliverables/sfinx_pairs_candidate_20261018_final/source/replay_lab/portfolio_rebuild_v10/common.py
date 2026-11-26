import copy, gzip, json
from pathlib import Path
import numpy as np
from ..competition_rebuild.common import load, covariance_path, write, stamp, sha, OUT as OLD
from ..macro_development.engine import simulate
from ..pairs_rl.verify import independent_accounting
from ..common_comparison.experiment import json_value

OUT=OLD.parents[1]/'portfolio_rebuild'/'20261013-v10'

def protocol():
    m,z,b,p=load()
    # These are research constraints, not claimed competition rules.
    p['risk'].update(gross_fraction=.6,net_fraction=.4,asset_fraction=.15,
        asset_notional=150000.,daily_volatility=.005,normal_turnover_fraction=.15,
        drawdown_halt=.08)
    return m,z,b,p

def allocate(raw,sigmas,times,daily_risk=.001,interval=24):
    """Same ex-ante risk budget, finite cap; no calibration on future PnL."""
    out=np.zeros_like(raw);held=np.zeros(raw.shape[1])
    for i in range(672,len(raw)):
        if times[i]%(interval*3600)==0:
            x=raw[i].copy()
            vol=np.sqrt(max(0.,24*x@sigmas[i]@x))
            if vol>1e-12:x*=daily_risk/vol
            else:x*=0
            x=np.clip(x,-.135,.135)
            x/=max(1.,abs(x).sum()/.54,abs(x.sum())/.36)
            held=x
        out[i]=held
    return out

def evaluate(m,p,sigma,targets,name,a,b,stage,phase,scenario='base',buffer=.2):
    r=simulate(m,dict(target_weight=targets),name,p,scenario,int(m.times[a]),int(m.times[b]),buffer=buffer,sigmas=sigma)
    audit=independent_accounting(r,m)
    assert audit['passed'] and r['gates']['accounting'] and r['gates']['information_time']
    dest=OUT/stage/'runs'/phase/f'{name}-{scenario}.json.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(dest,'wt') as f:json.dump(r,f,default=json_value,allow_nan=False)
    row={k:r[k] for k in ('policy','scenario','start','end','metrics','gates')}
    row.update(phase=phase,audit=audit,path=str(dest.relative_to(OUT)))
    write(dest.with_suffix('.summary.json'),row)
    print(phase,name,scenario,'ROI',round(r['metrics']['roi']*100,4),'STD',round(r['metrics']['daily_vol']*100,4),'DD',round(r['metrics']['max_drawdown']*100,4),'gate',all(r['gates'].values()),flush=True)
    return row

def phases(m):
    ix=lambda t:m.exact(int(np.datetime64(t,'s').astype('int64')))
    return {'development':(ix('2026-06-10T00:00:00'),ix('2026-07-29T00:00:00')),
       'later_reused':(ix('2026-07-29T00:00:00'),len(m.times)-1),
       'whole_reused':(ix('2026-06-10T00:00:00'),len(m.times)-1)}
