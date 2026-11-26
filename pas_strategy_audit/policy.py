"""Causal PAS trend candidate. Research target generation; no order client."""
import numpy as np
import pandas as pd


def intraday_targets(data):
    price=np.asarray(data['close'],float);times=np.asarray(data['times']);active=np.asarray(data['active'],bool)
    log=pd.DataFrame(np.log(price));fast=log.ewm(span=60,adjust=False,min_periods=60).mean().to_numpy()
    slow=log.ewm(span=360,adjust=False,min_periods=360).mean().to_numpy()
    roc=(log-log.shift(360)).to_numpy()
    count=pd.DataFrame(active.astype(float)).rolling(360,min_periods=360).sum().to_numpy()
    volatility=log.diff().rolling(360,min_periods=360).std(ddof=1).clip(lower=.0005/np.sqrt(60)).to_numpy()
    age=np.asarray(data['age_ms']);state=np.zeros(price.shape[1]);weights=np.zeros_like(state);out=np.zeros_like(price);reasons=[]
    for i,t in enumerate(times):
        if t%900000==0:
            ready=np.isfinite(slow[i])&np.isfinite(roc[i])&(age[i]<=900000)&(count[i]>=12)
            direction=np.sign(fast[i]-slow[i]);prior=state.copy()
            exit_=(state!=0)&ready&((state*direction<=0)|(state*(log.iloc[i].to_numpy()-slow[i])<=0))
            state[exit_]=0
            enter=ready&(state==0)&~exit_&(abs(roc[i])>.0038)&(direction*roc[i]>0)&(direction*(log.iloc[i].to_numpy()-slow[i])>0)
            state[enter]=direction[enter]
            # A stale market cannot justify an increase; replay also enforces freshness.
            raw=np.divide(state,np.maximum(volatility[i],1e-8),out=np.zeros_like(state),where=np.isfinite(volatility[i]))
            weights=raw/max(abs(raw).sum(),1.)
            reasons.append({'time':int(t),'state':state.tolist(),'changed':np.flatnonzero(state!=prior).tolist(),'eligible':ready.tolist(),'roc6h':np.nan_to_num(roc[i]).tolist()})
        out[i]=weights
    return out,reasons


def sticky_conflict_guard(base_events, regime_events, gross_weight_cap=.04):
    """Opposing observed PAS trend can only reduce targets until a new base source."""
    queue=[(x['time'],0,x) for x in base_events]+[(x['time'],1,x) for x in regime_events]
    base=None;regime=None;blocked=None;last=None;out=[]
    for time,kind,event in sorted(queue,key=lambda x:(x[0],x[1])):
        if kind==0:
            base=np.asarray(event['weights'],float).copy()
            # Budget first, then veto. Renormalizing after a veto would increase
            # the surviving assets and undo the intended exposure reduction.
            base/=max(1.,abs(base).sum()/gross_weight_cap)
            blocked=np.zeros_like(base,dtype=bool)
        else:
            regime=np.asarray(event['weights'],float)
        if base is None:continue
        if regime is not None:blocked |= base*regime<0
        target=np.where(blocked,0.,base)
        assert np.all(abs(target)<=abs(base)+1e-15)
        if last is None or not np.array_equal(target,last):
            out.append({'time':int(time),'weights':target.tolist(),'units':'weights'})
            last=target.copy()
    return out
