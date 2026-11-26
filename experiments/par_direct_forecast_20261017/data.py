import json,hashlib
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from .collect import HERE,OUT,RAW,ts

ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')

def prepare():
    cfg=json.loads((HERE/'protocol.json').read_text());p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    times=np.arange(ts(cfg['data_start']),ts(cfg['end'])+300,300);n=len(times);a=len(p['symbols'])
    result={'times':times};coverage=[]
    for venue in ('par','reference'):
        close=np.full((n,a),np.nan);vol=np.zeros((n,a));trades=vol.copy()
        for j,symbol in enumerate(p['symbols']):
            rows=json.loads((RAW/f'{venue}_{symbol}_5m.json').read_text())['response']
            lookup={int(r['T']+1)//1000:r for r in rows}
            for i,t in enumerate(times):
                if int(t) not in lookup:raise ValueError(f'Missing {venue} {symbol} {t}')
                r=lookup[int(t)];assert int(r['T'])==int(r['t'])+299999
                close[i,j]=float(r['c']);vol[i,j]=float(r['v']);trades[i,j]=int(r['n'])
            coverage.append(dict(venue=venue,symbol=symbol,observations=n,active_bars=int((vol[:,j]>0).sum())))
        result[venue+'_close']=close;result[venue+'_volume']=vol;result[venue+'_trades']=trades
    with np.load(ROOT/'experiments/historical_par_20261017/artifacts/inputs.npz') as z:old={k:z[k].copy() for k in z.files}
    ix=np.searchsorted(old['times'],times,side='right')-1
    assert np.all(ix>=0) and np.all(old['times'][ix]<=times)
    for k in ('first','second','retention','sigma'):result[k]=old[k][ix]
    result['signal_time']=old['times'][ix]
    result['funding']=np.zeros((n,a));result['funding_observed']=np.zeros((n,a),bool)
    result['known_hourly_funding']=np.zeros((n,a))
    for i,t in enumerate(times):
        j=np.searchsorted(old['funding_times'],t,side='left')-1
        v=old['funding_history'][max(0,j-23):j+1]
        good=np.isfinite(v).all(0)&(len(v)==24)
        result['known_hourly_funding'][i]=np.where(good,.5*v[-1]+.5*np.nanmean(v,axis=0),0.)
        if t%3600==0:
            f=old['funding_history'][np.searchsorted(old['funding_times'],t)]
            assert np.isfinite(f).all();result['funding'][i]=f;result['funding_observed'][i]=True
    np.savez_compressed(OUT/'inputs.npz',**result)
    save(OUT/'data_manifest.json',dict(coverage=coverage,observations=n,
        historical_candle_coverage_complete=True,historical_oracle_or_book=False,
        sources={str(q):sha(q) for q in RAW.glob('*.json')},orders_sent=0))
    return result

if __name__=='__main__':prepare()
