"""Symmetric equal-notional relative trend and rolling AR(1) forecasts.

No asset labels are inputs. Reversing pair order reverses its signal. The
hedge is dollar neutral at formation, not claimed market-beta neutral.
"""
import numpy as np
import pandas as pd
from auto_pair_selection.selection import IDS


def relative_features(d):
    t=d['market_times'];logs=np.log(d['close']);r=pd.DataFrame(logs).diff().to_numpy()
    n=len(t);q=len(IDS)
    s=np.column_stack([logs[:,a]-logs[:,b] for a,b in IDS])
    rr=np.column_stack([(r[:,a]-r[:,b])/2 for a,b in IDS])
    vol=pd.DataFrame(rr).rolling(168,min_periods=168).std(ddof=0).to_numpy()
    fresh=np.column_stack([(d['volume'][:,a]>0)&(d['volume'][:,b]>0)&(d['trades'][:,a]>0)&(d['trades'][:,b]>0) for a,b in IDS])
    long=np.full((n,q),np.nan);short=long.copy()
    long[168:]=(s[168:]-s[:-168])/2;short[24:]=(s[24:]-s[:-24])/2
    valid=np.isfinite(vol)&(vol>1e-6)&np.isfinite(long)&np.isfinite(short)&fresh
    return dict(spread=s,vol=vol,fresh=fresh,valid=valid,week=long,day=short)


def ar_fit(series,fresh,source_index,window=720):
    start=max(1,source_index-window+1)
    ix=np.arange(start,source_index+1)
    valid=fresh[ix]&fresh[ix-1]&np.isfinite(series[ix])&np.isfinite(series[ix-1])
    ix=ix[valid]
    if len(ix)<168:return None
    x,y=series[ix-1],series[ix];dx=x-x.mean();dy=y-y.mean()
    den=float(dx@dx)
    if den<1e-12:return None
    rho=float(dx@dy/den)
    if not 0<rho<1:return None
    half_life=float(-np.log(2)/np.log(rho))
    if not 6<=half_life<=336:return None
    intercept=float(y.mean()-rho*x.mean())
    return dict(rho=rho,equilibrium=intercept/(1-rho),half_life=half_life,
                observations=len(ix),last_price_index=int(ix[-1]))


def forecasts(d,horizon,kind):
    f=relative_features(d);t=d['market_times'];mu=np.full_like(f['vol'],np.nan)
    valid=f['valid'];trend=f['week']*(horizon/168)
    aligned=(np.sign(f['week'])==np.sign(f['day']))
    records=[];models={}
    if kind not in ('trend','reversion','regime'):raise ValueError('Unknown relative model')
    for i,now in enumerate(t):
        if now%86400==0 and kind!='trend':
            models={}
            for k in range(len(IDS)):
                m=ar_fit(f['spread'][:,k],f['fresh'][:,k],i)
                if m is not None:models[k]=m;records.append(dict(source=int(now),pair_index=k,**m))
        for k in np.flatnonzero(valid[i]):
            mr=None
            if k in models:
                m=models[k];mr=(m['equilibrium']-f['spread'][i,k])*(1-m['rho']**horizon)/2
            if kind=='trend':value=trend[i,k]
            elif kind=='reversion':value=mr
            else:value=trend[i,k] if aligned[i,k] else mr
            if value is not None:
                mu[i,k]=np.clip(value,-4*f['vol'][i,k]*np.sqrt(horizon),4*f['vol'][i,k]*np.sqrt(horizon))
    entry=np.isfinite(mu)&valid
    if kind=='trend':entry&=aligned
    return mu,f,entry,records
