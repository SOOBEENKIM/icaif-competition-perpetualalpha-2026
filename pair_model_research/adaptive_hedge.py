"""Symmetric major-axis price hedge and local AR relative-value model.

Estimated AR t statistics are screening diagnostics, not valid 45-way
cointegration p-values. Backtests are needed; no stationarity proof is claimed.
"""
import numpy as np
from auto_pair_selection.selection import IDS


def fit_pair(logs,active,index,ids,window=1440):
    start=max(0,index-window+1);ix=np.arange(start,index+1);a,b=ids
    good=active[ix,a]&active[ix,b];ii=ix[good]
    if len(ii)<.8*window:return None
    x,y=logs[ii,a],logs[ii,b];dx=x-x.mean();dy=y-y.mean();vx=float(dx@dx);vy=float(dy@dy);cov=float(dx@dy)
    if min(vx,vy)<=1e-16 or cov<=0:return None
    beta=float(np.sqrt(vx/vy)) # symmetric standardized-major-axis hedge, not OLS
    if not .1<=beta<=10:return None # retained executable contract, no clipping
    spread=logs[:,a]-beta*logs[:,b]
    rows=ii[ii>start];rows=rows[active[rows-1,a]&active[rows-1,b]]
    if len(rows)<.8*window-1:return None
    xx=spread[rows-1];yy=spread[rows];xc=xx-xx.mean();yc=yy-yy.mean();den=float(xc@xc)
    if den<1e-16:return None
    rho=float(xc@yc/den)
    if not 0<rho<1:return None
    half=float(-np.log(2)/np.log(rho))
    if not 6<=half<=336:return None
    intercept=float(yy.mean()-rho*xx.mean());res=yy-intercept-rho*xx
    se=float(np.sqrt((res@res)/max(1,len(rows)-2)/den));tstat=(rho-1)/se if se>0 else -1e6
    rv=(np.diff(logs[max(0,index-168):index+1,a])-beta*np.diff(logs[max(0,index-168):index+1,b]))/(1+beta)
    vol=float(np.std(rv,ddof=0))
    if not np.isfinite(vol) or vol<=1e-6:return None
    return dict(beta=beta,rho=rho,equilibrium=intercept/(1-rho),half_life=half,ar_tstat=float(tstat),vol=vol,
                rows=len(rows),last_price_index=int(rows[-1]),first_price_index=int(rows[0]-1),level_correlation=cov/np.sqrt(vx*vy))


def forecasts(d,start,horizon=72):
    times=d['market_times'];logs=np.log(d['close']);active=(d['volume']>0)&(d['trades']>0);shape=(len(times),len(IDS))
    mu=np.full(shape,np.nan);beta=mu.copy();vol=mu.copy();strong=np.zeros(shape,bool);records=[];models={}
    for i,now in enumerate(times):
        if now<start:continue
        if now%86400==0:
            models={}
            for k,ids in enumerate(IDS):
                model=fit_pair(logs,active,i,ids);records.append(dict(source=int(now),pair_index=k,model=model))
                if model is not None:models[k]=model
        for k,m in models.items():
            a,b=IDS[k]
            if not active[i,a] or not active[i,b]:continue
            s=logs[i,a]-m['beta']*logs[i,b];v=(m['equilibrium']-s)*(1-m['rho']**horizon)/(1+m['beta'])
            mu[i,k]=np.clip(v,-4*m['vol']*np.sqrt(horizon),4*m['vol']*np.sqrt(horizon));beta[i,k]=m['beta'];vol[i,k]=m['vol'];strong[i,k]=m['ar_tstat'] < -3.
    return mu,beta,vol,strong,records
