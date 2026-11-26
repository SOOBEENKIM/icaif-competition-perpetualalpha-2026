"""Separate local-price basis dynamics from unforecastable reference movement.

AR(1) fits use pairs of consecutive actively traded bars strictly before the
fit time. Forecasted return begins after one-bar arrival, so already-decayed
basis is not credited as an executable gain. This is not full cointegration.
"""
import numpy as np
from .data import ts

def predict(d,de,cfg):
    times=d['times'];basis=np.log(d['par_close']/d['reference_close'])
    active=(d['par_volume']>0)&(d['par_trades']>0)
    out={f'basis_ar_{h}':np.full(de['beta'].shape,np.nan) for h in cfg['horizons_hours']};records=[]
    dates=[ts(s) for s in cfg['fit_dates']]
    for fi,at in enumerate(dates):
        until=dates[fi+1] if fi+1<len(dates) else int(times[-1])+1;pars={}
        for j in range(basis.shape[1]):
            mask=(times[1:]<at)&(times[:-1]>=at-7*86400)&active[:-1,j]&active[1:,j]
            ids=np.flatnonzero(mask)
            if len(ids)<30:
                records.append(dict(fit_time=at,asset=j,rows=len(ids),status='insufficient_consecutive_active_bars'));continue
            x,y=basis[ids,j],basis[ids+1,j];v=np.var(x)
            if v<1e-12:continue
            phi=float(np.clip(np.mean((x-x.mean())*(y-y.mean()))/v,0.,.999))
            mu=float(np.median(basis[np.flatnonzero((times<at)&(times>=at-7*86400)&active[:,j]),j]))
            pars[j]=(mu,phi)
            records.append(dict(fit_time=at,asset=j,rows=len(ids),last_label_time=int(times[ids[-1]+1]),mean=mu,phi=phi,
                                half_life_minutes=float(-5*np.log(2)/np.log(phi)) if phi>0 else 0.))
        ix=np.flatnonzero((times>=at)&(times<until))
        for k,(a,b) in enumerate(de['pairs']):
            if a not in pars or b not in pars:continue
            for h in cfg['horizons_hours']:
                changes=[]
                for j in (a,b):
                    mu,phi=pars[j]
                    expected_entry=mu+phi*(basis[ix,j]-mu)
                    expected_exit=mu+phi**(h*12+1)*(basis[ix,j]-mu)
                    changes.append(np.expm1(expected_exit-expected_entry))
                be=de['beta'][ix,k];out[f'basis_ar_{h}'][ix,k]=(changes[0]-be*changes[1])/(1+be)
    return out,records
