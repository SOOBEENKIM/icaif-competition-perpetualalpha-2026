"""EMA entry with explicit monotone trailing exits and cost-aware re-entry.

Second development family: position lifecycle rather than repeated expert
switching. Uses completed candles, fixed risk sleeves and no learned labels.
"""
from dataclasses import dataclass,asdict
import hashlib,json
import numpy as np
from ..paper_reproductions.ema_wf.model import ema
from .model import rolling_mean


@dataclass(frozen=True)
class Policy:
    frame:int=24
    fast:int=5
    slow:int=40
    confidence:float=.4
    atr_multiple:float=3.
    breakout:int=0
    reentry:int=0
    exit_cross:bool=False
    buffer:float=.2
    decision_hours:int=6
    scale:float=1.

    @property
    def key(self):return 'ema-cycle-'+hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:12]


def features(a):
    n=len(a['known']);close=a['close'][:,:3];banks={}
    for frame in (6,24):
        ids=np.flatnonzero(a['known']%(frame*3600)==0)
        positions=np.searchsorted(ids,np.arange(n),side='right')-1
        prices=close[ids];matrix={}
        for k in (3,5,10,20,40,60):
            ev=np.column_stack([ema(prices[:,j],k) for j in range(3)])
            v=np.full((n,3),np.nan);v[positions>=0]=ev[positions[positions>=0]];matrix[k]=v
        banks[frame]=matrix
    ids=np.flatnonzero(a['known']%86400==0);last=-1;daily=[];highs=[];lows=[]
    atr=np.full((n,3),np.nan);eff=np.zeros((n,3));upper={k:np.full((n,3),np.nan) for k in (3,5)}
    for i,t in enumerate(a['known']):
        if t%86400==0:
            current=close[i];daily.append(current)
            hi=a['high'][last+1:i+1,:3].max(0);lo=a['low'][last+1:i+1,:3].min(0)
            old=daily[-2] if len(daily)>1 else current
            highs.append(np.maximum(hi-lo,np.maximum(abs(hi-old),abs(lo-old))));last=i
            x=np.array(daily)
            if len(x)>=20:current_atr=np.mean(highs[-20:],axis=0)
            else:current_atr=np.full(3,np.nan)
            if len(x)>=8:
                d=np.diff(np.log(x[-8:]),axis=0);current_eff=d.sum(0)/np.maximum(abs(d).sum(0),1e-12)
            else:current_eff=np.zeros(3)
            for h in upper:
                if len(x)>h:upper[h][i]=x[-h-1:-1].max(0)
            atr[i]=current_atr;eff[i]=current_eff
        elif i:
            atr[i]=atr[i-1];eff[i]=eff[i-1]
            for h in upper:upper[h][i]=upper[h][i-1]
    return dict(ema=banks,atr=atr,eff=eff,upper=upper)


def fit(a,p,volatility,detail=True,f=None):
    if f is None:f=features(a)
    n=len(a['known']);signal=np.zeros((n,10));stops=np.full(3,-np.inf);state=np.zeros(3)
    blocked=np.zeros(3,dtype=bool);reentry=np.zeros(3);prior_exit=np.zeros(3,dtype=np.int64)
    rows=[];last=np.zeros(3);last_decision=0;decisions=np.zeros(n,dtype=np.int64)
    fast=f['ema'][p.frame][p.fast];slow=f['ema'][p.frame][p.slow]
    for i,t in enumerate(a['known']):
        if t%(p.decision_hours*3600)==0:
            price=a['close'][i,:3];atr=f['atr'][i];ready=np.isfinite(atr)&np.isfinite(slow[i])
            up=(fast[i]>slow[i])&(price>fast[i]*1.0038)&(f['eff'][i]>=p.confidence)&ready
            if p.breakout:up&=price>=f['upper'][p.breakout][i]
            if p.reentry:
                blocked&=(price<=reentry)&(fast[i]>slow[i])
                up&=~blocked
            existing=state>0
            stops=np.where(existing,np.maximum(stops,price-p.atr_multiple*atr),stops)
            exits=existing&((price<=stops)|((fast[i]<slow[i])&p.exit_cross))
            state[exits]=0;blocked[exits]=True;reentry[exits]=price[exits]*1.0038;prior_exit[exits]=t
            enter=up&(state==0)&~exits&(t-prior_exit>=24*3600)
            state[enter]=1;stops[enter]=price[enter]-p.atr_multiple*atr[enter]
            if detail and (np.any(state!=last) or t%86400==0):
                for j in range(3):
                    rows.append(dict(asset=j,cutoff=int(t),fast=p.fast,slow=p.slow,frame=p.frame,
                        stop=float(stops[j]) if np.isfinite(stops[j]) else None,
                        atr=float(atr[j]) if np.isfinite(atr[j]) else None,
                        efficiency=float(f['eff'][i,j]),enabled=bool(state[j]),
                        entered=bool(enter[j]),exited=bool(exits[j])))
            last=state.copy();last_decision=int(t)
        signal[i,:3]=state;decisions[i]=last_decision
    budget=np.where(np.isfinite(volatility),np.minimum(.25/np.maximum(volatility,.01),2.)*.01,0.)
    return dict(signal=signal,target_weight=signal*budget*p.scale,volatility=volatility.copy(),decision=decisions),rows
