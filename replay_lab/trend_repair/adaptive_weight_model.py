"""Learn the mixture of four trend horizons from matured net expert outcomes.

Equal 5/10/20/30-day channel weights are a modelling choice, not a constraint.
An entropy-regularized online learner tilts toward horizons that have recently
retained return after turnover cost. Each horizon stays between 10% and 40%.
The full per-asset risk sleeve and all account limits remain unchanged.
"""
from dataclasses import dataclass
import numpy as np
from .refinement import RefinementPolicy,fit as original


@dataclass(frozen=True)
class AdaptiveWeightPolicy:
    model:str='control'
    days:int=14
    strength:float=1.
    buffer:float=.2

    @property
    def key(self):return f'adaptive-horizons-{self.model}-d{self.days}-s{self.strength:g}'.replace('.','p')


def bounded_weights(scores):
    scores=np.clip(np.asarray(scores,dtype=float),-3,3)
    p=np.exp(scores-scores.max());p/=p.sum()
    # Entropy regularization around equal allocation and bounded-simplex
    # projection prevent one noisy short-history expert from taking over.
    target=.5*np.full(4,.25)+.5*p
    lo,hi=-1.,1.
    for _ in range(60):
        offset=(lo+hi)/2
        if np.clip(target-offset,.1,.4).sum()>1:lo=offset
        else:hi=offset
    return np.clip(target-(lo+hi)/2,.1,.4)


def fit(a,p,f,forecast):
    m,states=original(a,RefinementPolicy(),f,forecast)
    by_time={s['known']:s for s in states};n=len(a['known']);target=np.zeros((n,10))
    weights=np.full(4,.25);enabled=np.zeros((4,3));proxy=[];checkpoints=[];previous=None;rows=[]
    last_bucket=None;trend_start=None
    for i,known in enumerate(a['known']):
        if known%86400==0 and int(known) in by_time and np.isfinite(m['volatility'][i,:3]).all():
            sig=np.array(by_time[int(known)]['enabled'],dtype=float)
            sigma=m['volatility'][i,:3]
            scale=np.minimum(.25/np.maximum(sigma,.01),2)*.01
            exposure=sig*scale[None,:]*(m['signal'][i,:3]>0)[None,:]
            if previous is not None:
                # One-day expert label ends at THIS completed close. Approximate
                # execution costs are for training only; final portfolio uses
                # the full common broker, volume constraints and funding.
                ret=a['close'][i,:3]/previous['open']-1
                pnl=(previous['exposure']*ret[None,:]).sum(1)-previous['cost']
                proxy.append(dict(maturity=int(known),return_=pnl.tolist()))
            drift=previous['exposure']*(a['close'][i,:3]/previous['open'])[None,:] if previous else np.zeros((4,3))
            cost=abs(exposure-drift).sum(1)*.0019
            next_open=a['open'][i+1,:3] if i+1<n else a['close'][i,:3]
            previous=dict(exposure=exposure,open=next_open,cost=cost)
            bucket=int(known//86400)
            if np.any(m['signal'][i,:3]>0):
                if trend_start is None:trend_start=int(known)
            else:trend_start=None
            if p.model=='slow':weights=np.array([.1,.2,.3,.4])
            elif p.model=='fast':weights=np.array([.4,.3,.2,.1])
            elif p.model=='lifecycle':
                age=0 if trend_start is None else (known-trend_start)/86400
                z=min(1.,age/14)
                weights=(1-z)*np.array([.4,.3,.2,.1])+z*np.array([.1,.2,.3,.4])
            elif p.model=='learned' and bucket!=last_bucket:
                eligible=[x for x in proxy if known-p.days*86400<=x['maturity']<known]
                if len(eligible)>=7:
                    y=np.array([x['return_'] for x in eligible]);k=len(y)
                    pooled=max(float(np.sqrt(np.mean(np.var(y,axis=0,ddof=1)))),1e-6)
                    scores=p.strength*y.mean(0)/pooled*np.sqrt(k)
                    weights=bounded_weights(scores)
                    checkpoints.append(dict(cutoff=int(known),last_maturity=eligible[-1]['maturity'],
                        days=k,mean_net_daily_return=y.mean(0).tolist(),pooled_sigma=pooled,
                        scores=scores.tolist(),weights=weights.tolist()))
                last_bucket=bucket
            elif p.model not in ('control','slow','fast','lifecycle','learned'):raise ValueError('Unknown channel weighting')
            enabled=sig
            rows.append(dict(known=int(known),weights=weights.tolist(),signal=m['signal'][i,:3].tolist()))
        sigma=m['volatility'][i,:3]
        if np.isfinite(sigma).all():
            weighted_signal=(weights[:,None]*enabled).sum(0)*(m['signal'][i,:3]>0)
            target[i,:3]=weighted_signal*np.minimum(.25/np.maximum(sigma,.01),2)*.01
    if p.model!='control':m['target_weight']=target
    return m,dict(states=rows,checkpoints=checkpoints,expert_daily_returns=proxy)
