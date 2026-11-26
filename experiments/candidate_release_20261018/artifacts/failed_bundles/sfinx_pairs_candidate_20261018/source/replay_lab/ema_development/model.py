"""Cost-aware, hysteretic EMA experts. Historical inputs only; no exchange API.

An extension, not a reproduction, of the already reproduced EMA paper. Entry
and reversal require a cost/volatility band and completed-bar confirmation.
All expert outcomes mature before they enter the rolling parameter selector.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
from ..paper_reproductions.ema_wf.model import ema, PAIRS


@dataclass(frozen=True)
class Policy:
    frame: int = 6
    band: float = 1.
    confirmation: int = 2
    confidence: float = .2
    short: float = 0.
    train_days: int = 28
    deploy_days: int = 7
    blend: int = 3
    selector: str = 'positive'
    scale: float = 1.
    buffer: float = .2
    decision_hours: int = 6

    @property
    def key(self):
        return 'ema-cost-'+hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:12]


def rolling_mean(x, hours):
    out=np.full_like(x,np.nan,dtype=float)
    s=np.concatenate([np.zeros((1,)+x.shape[1:]),np.cumsum(x,axis=0)])
    out[hours-1:] = (s[hours:]-s[:-hours])/hours
    return out


def features(a):
    n=len(a['times']);close=a['close'][:,:3]
    prev=np.vstack([close[:1],close[:-1]])
    tr=np.maximum(a['high'][:,:3]-a['low'][:,:3],
        np.maximum(abs(a['high'][:,:3]-prev),abs(a['low'][:,:3]-prev)))/close
    atr=rolling_mean(tr,24)
    confidence=np.zeros((n,3));days=[];last=np.zeros(3)
    for i,t in enumerate(a['known']):
        if t%86400==0:
            days.append(close[i].copy())
            if len(days)>=8:
                log=np.log(np.array(days[-8:]));r=np.diff(log,axis=0)
                last=(log[-1]-log[0])/np.maximum(abs(r).sum(0),1e-12)
        confidence[i]=last
    banks={}
    for frame in (1,6,24):
        ids=np.flatnonzero(a['known']%(3600*frame)==0)
        # These are EMAs of completed resampled closes, not an EMA using an
        # unfinished daily candle. The last completed value is carried forward.
        periods=sorted({k for pair in PAIRS for k in pair})
        e={k:np.column_stack([ema(close[ids,j],k) for j in range(3)]) for k in periods}
        gap=np.stack([(e[f]-e[s])/close[ids] for f,s in PAIRS],axis=2)
        full=np.full((n,3,len(PAIRS)),np.nan)
        loc=np.searchsorted(ids,np.arange(n),side='right')-1;valid=loc>=0
        full[valid]=gap[loc[valid]];banks[frame]=full
    return dict(atr=atr,confidence=confidence,gap=banks)


def expert_signals(a,p,f):
    gap=f['gap'][p.frame];n=len(gap);state=np.zeros((3,len(PAIRS)))
    requested=np.zeros_like(state);count=np.zeros_like(state,dtype=int)
    signals=np.zeros_like(gap)
    # Minimum entry separation covers the assumed 19bp/side round trip.
    for i,t in enumerate(a['known']):
        if t%(p.decision_hours*3600)==0:
            width=np.maximum(.0038,p.band*f['atr'][i]*np.sqrt(p.frame))[:,None]
            g=gap[i];conf=f['confidence'][i,:,None]
            desired=state.copy()
            # Neutral band: retain direction until the cross reverses. A new
            # position requires the stronger outer band, eliminating chatter.
            desired[(state>0)&(g<0)]=0
            desired[(state<0)&(g>0)]=0
            desired[(g>width)&(conf>=p.confidence)]=1
            if p.short:
                desired[(g<-width)&(conf<=-p.confidence)]=-p.short
            desired[~np.isfinite(g)]=0
            same=desired==requested
            count=np.where(same,count+1,1);requested=desired
            ready=count>=p.confirmation
            state=np.where(ready,requested,state)
        signals[i]=state
    return signals


def select_model(a,p,volatility,signals,detail=False):
    n=len(a['times']);weights=np.zeros((n,10));signal=np.zeros_like(weights)
    close=a['close'][:,:3];returns=np.vstack([np.zeros((1,3)),close[1:]/close[:-1]-1])
    previous=np.concatenate([np.zeros_like(signals[:1]),signals[:-1]])
    # Conservative proxy for parameter selection. Actual ledgers use broker
    # prices, funding and liquidity; this proxy is never called a backtest.
    pnl=previous*returns[:,:,None]-.0019*abs(signals-previous)
    active=(abs(previous)>0).astype(float)
    cs=np.concatenate([np.zeros_like(pnl[:1]),np.cumsum(pnl,axis=0)])
    ss=np.concatenate([np.zeros_like(pnl[:1]),np.cumsum(pnl*pnl,axis=0)])
    ac=np.concatenate([np.zeros_like(active[:1]),np.cumsum(active,axis=0)])
    chosen=[np.array([],dtype=int) for _ in range(3)];states=[];last_bucket=None
    anchor=int(a['known'][0]//86400);training=p.train_days*24
    budget=np.where(np.isfinite(volatility),np.minimum(.25/np.maximum(volatility,.01),2.)*.01,0.)
    for i,t in enumerate(a['known']):
        bucket=(int(t//86400)-anchor)//p.deploy_days
        if i>=training and t%86400==0 and bucket!=last_bucket:
            lo=i-training+1;count=training
            mean=(cs[i+1]-cs[lo])/count
            variance=np.maximum((ss[i+1]-ss[lo])/count-mean*mean,1e-12)
            score=mean/np.sqrt(variance)
            # A losing expert is not forced into the portfolio. Require >=24
            # active hourly returns to avoid selecting a one-trade anomaly.
            valid=(ac[i+1]-ac[lo]>=24)&(mean>0)
            for j in range(3):
                ids=np.flatnonzero(valid[j])
                if p.selector=='equal':
                    chosen[j]=np.flatnonzero(np.isfinite(features_ready(a,p,i,j)))
                else:
                    chosen[j]=ids[np.argsort(-score[j,ids],kind='stable')[:p.blend]]
                if detail:
                    states.append(dict(asset=j,cutoff=int(t),training_start=int(a['known'][lo]),
                        training_last=int(t),next_refit=int(t)+p.deploy_days*86400,
                        experts=[dict(fast=PAIRS[k][0],slow=PAIRS[k][1],frame=p.frame,
                            net_hourly_mean=float(mean[j,k]),score=float(score[j,k])) for k in chosen[j]]))
            last_bucket=bucket
        for j in range(3):
            if len(chosen[j]):signal[i,j]=np.mean(signals[i,j,chosen[j]])
        weights[i]=signal[i]*budget[i]*p.scale
    return dict(signal=signal,target_weight=weights,volatility=volatility.copy(),decision=a['known'].copy()),states


def features_ready(a,p,i,j):
    # An expert needs slow completed bars. Length is determined using only
    # timestamps up to this decision; no data-end dependence or price lookahead.
    completed=int(np.count_nonzero(a['known'][:i+1]%(p.frame*3600)==0))
    return np.array([1. if completed>=s else np.nan for _,s in PAIRS])


def fit(a,p,volatility,detail=True):
    f=features(a)
    return select_model(a,p,volatility,expert_signals(a,p,f),detail)
