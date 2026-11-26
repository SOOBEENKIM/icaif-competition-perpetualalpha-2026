"""Calibrate the successful research AR prior without replacing its hedge.

Unlike choosing an expert by its in-sample AR fit, this learns a scale from
strictly matured returns of forecasts generated at their original source.
"""
import numpy as np
from pair_model_research.adaptive_hedge import forecasts
from auto_pair_selection.selection import IDS


def coefficients(t,mu,unit,y,fresh,source):
    # Fixed72h spacing avoids counting overlapping hourly forecasts as
    # independent calibration origins. Both legs use source-time beta labels.
    mask=(t>=source-90*86400)&(t+72*3600<source)&(t%(72*3600)==0)
    ii=np.flatnonzero(mask);good=np.isfinite(mu[ii])&np.isfinite(y[ii])&fresh[ii]&(unit[ii]>0)
    x=np.divide(mu[ii],unit[ii],out=np.zeros_like(mu[ii]),where=unit[ii]>0)
    yy=np.divide(y[ii],unit[ii],out=np.zeros_like(y[ii]),where=unit[ii]>0)
    x=np.where(good,np.clip(x,-4,4),0.);yy=np.where(good,np.clip(yy,-4,4),0.)
    dateweight=np.exp2(-(source-(t[ii]+72*3600))/(30*86400))
    weights=good*dateweight[:,None]/np.maximum(1,good.sum(1))[:,None]
    global_=float(np.clip(np.sum(weights*x*yy)/(np.sum(weights*x*x)+6),0,1))
    local=np.zeros(mu.shape[1]);records=[]
    for k in range(mu.shape[1]):
        g=good[:,k];w=dateweight[g];xx=x[g,k];target=yy[g,k]
        slope=float(np.clip((np.sum(w*xx*target)+6*global_)/(np.sum(w*xx*xx)+6),0,1)) if g.sum()>=8 else global_
        local[k]=slope;records.append(dict(pair_index=k,matured_origins=int(g.sum()),slope=slope,last_maturity=int(max(t[ii[g]])+72*3600) if g.any() else None))
    return global_,local,records


def build(d,start):
    t=d['market_times'];mu,beta,vol,_,formation=forecasts(d,int(t[0]+1440*3600),72)
    unit=vol*np.sqrt(72);y=np.full_like(mu,np.nan);active=(d['volume']>0)&(d['trades']>0)
    fresh=np.array([active[:,a]&active[:,b] for a,b in IDS]).T;terminal=np.zeros_like(fresh);terminal[:-72]=fresh[72:]
    for k,(a,b) in enumerate(IDS):
        rr=d['close'][72:,[a,b]]/d['close'][:-72,[a,b]]-1;b0=beta[:-72,k]
        y[:-72,k]=(rr[:,0]-b0*rr[:,1])/(1+b0)
    means={k:np.full_like(mu,np.nan) for k in ['pooled_sma_72','local_sma_72']};records=[]
    for source in np.unique(t[t>=start]//86400*86400):
        ag,al,audit=coefficients(t,mu,unit,y,fresh&terminal,int(source));ii=np.flatnonzero((t>=source)&(t<source+86400))
        means['pooled_sma_72'][ii]=mu[ii]*ag;means['local_sma_72'][ii]=mu[ii]*al
        records.append(dict(source=int(source),global_coefficient=ag,pairs=audit))
    states={}
    for arm,mean in means.items():
        available=np.isfinite(mean)&fresh
        states[arm]=dict(times=t,dense=True,pairs=np.asarray(IDS),signal_times=np.broadcast_to(t[:,None],mean.shape),source_prices=d['close'],forecast_times=t,beta=beta,vol=vol,mu=mean,
            directions=available.astype(int),economic_available=available,economic_action=np.where(available&(abs(mean)>.0038),np.sign(np.nan_to_num(mean)),0).astype(int))
    return states,records,formation
