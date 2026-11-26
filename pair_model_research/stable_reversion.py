"""All-pair AR hypotheses with disjoint past-window agreement.

Two fitted means agreeing is a robustness heuristic, not a stationarity test.
The previous block is evaluated at the CURRENT spread, without future prices.
"""
import numpy as np
from .relative import relative_features,ar_fit
from auto_pair_selection.selection import IDS


def forecasts(d,horizon=72):
    f=relative_features(d);t=d['market_times'];mu=np.full_like(f['vol'],np.nan);entry=np.zeros_like(f['fresh']);records=[];current={};previous={}
    for i,now in enumerate(t):
        if now%86400==0:
            current={};previous={}
            for k in range(len(IDS)):
                a=ar_fit(f['spread'][:,k],f['fresh'][:,k],i)
                b=ar_fit(f['spread'][:,k],f['fresh'][:,k],i-720) if i>=720 else None
                if a is not None:current[k]=a
                if b is not None:previous[k]=b
                records.append(dict(source=int(now),pair_index=k,current=a,previous=b))
        for k in np.flatnonzero(f['valid'][i]):
            if k not in current:continue
            a=current[k];s=f['spread'][i,k];v=(a['equilibrium']-s)*(1-a['rho']**horizon)/2
            # Preserve a usable current forecast for inventory management,
            # even when an entry has no confirming previous-window estimate.
            mu[i,k]=np.clip(v,-4*f['vol'][i,k]*np.sqrt(horizon),4*f['vol'][i,k]*np.sqrt(horizon))
            if k not in previous:continue
            b=previous[k];prior=(b['equilibrium']-s)*(1-b['rho']**horizon)/2
            entry[i,k]=np.sign(prior)==np.sign(v) and min(abs(prior),abs(v))>.0043
    return mu,f,entry,records
