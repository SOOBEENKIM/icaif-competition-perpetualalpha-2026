"""Cost-separated deviations from a trailing mean, with bounded trade life."""
import numpy as np
import pandas as pd
from ..portfolio_rebuild_v10.lifecycle_allocation import targets
from ..alpha_rebuild.common import limits

def features(m,days):
    log=pd.DataFrame(np.log(m.close));hours=24*days
    mean=log.shift(1).rolling(hours).mean().to_numpy()
    sd=log.shift(1).rolling(hours).std(ddof=0).clip(lower=.001).to_numpy()
    z=(log.to_numpy()-mean)/sd
    ret=log.diff().fillna(0)
    vol=ret.ewm(span=168,adjust=False,min_periods=168).std().clip(lower=.0005).to_numpy()*np.sqrt(24)
    funding=pd.DataFrame(m.funding).shift(1).rolling(24).mean().fillna(0).to_numpy()
    fast=log.ewm(span=240,adjust=False,min_periods=240).mean().to_numpy()
    slow=log.ewm(span=1440,adjust=False,min_periods=1440).mean().to_numpy()
    return dict(mean=mean,sd=sd,z=z,vol=vol,funding=funding,trend=np.sign(fast-slow))

def signal(m,f,days,universe,regime):
    out=np.zeros_like(m.close);held=np.zeros(10);started=np.zeros(10);exited=np.full(10,-10**12);entry=np.zeros(10);stop=np.zeros(10)
    allowed=np.arange(10)<3 if universe=='crypto' else np.ones(10,bool);horizon=min(days*48,168)
    for i,t in enumerate(m.times):
        if i>=1440 and t%21600==0:
            logp=np.log(m.close[i]);zz=f['z'][i];sd=f['sd'][i]
            # A recovery or thesis failure closes the position; never flip in
            # the same decision. Funding forecasts use only settled prior hours.
            exit=(held!=0)&((held*zz>=-.1)|(held*(logp-entry)<=-stop)|(t-started>=horizon*3600))
            held[exit]=0;exited[exit]=t
            side=-np.sign(zz);distance=abs(logp-f['mean'][i])
            cost=.0038+np.maximum(0,side*f['funding'][i])*horizon
            enter=allowed&(held==0)&~exit&(t-exited>=86400)&(abs(zz)>=2)&(abs(zz)<4)&(distance>cost)&(m.volume[i]>0)&(m.trades[i]>0)
            if regime=='aligned':enter&=side*f['trend'][i]>0
            held[enter]=side[enter];entry[enter]=logp[enter];stop[enter]=2*sd[enter];started[enter]=t
        out[i]=held
    return out

def build(m,sigma,base):
    weights={};specs={}
    for days in (1,3,7):
        f=features(m,days)
        for universe in ('crypto','all'):
            for regime in ('unconditional','aligned'):
                s=signal(m,f,days,universe,regime);raw=np.nan_to_num(s/np.maximum(f['vol'],.003),nan=0)
                w=.9*targets(raw,sigma,m.times,days=3,risk=.001)
                for blend in (0.,.25,.5):
                    name=f'{universe}_d{days}_{regime}_'+('direct' if blend==0 else f'add{int(blend*100)}')
                    weights[name]=w if blend==0 else limits(base+blend*w)
                    specs[name]=dict(days=days,universe=universe,regime=regime,blend=blend)
    return weights,specs
