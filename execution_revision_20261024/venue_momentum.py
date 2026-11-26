"""Fixed multi-horizon PAS trend with cost-aware entry and volatility sizing.

No fit on the competition's outcomes. Horizons (6h,24h,72h), 24h valuation
horizon, and existing portfolio risk budget are fixed before this experiment.
The average extrapolated drift is a hypothesis, not a calibrated forecast.
"""
import numpy as np
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

def features(t,price_times,history,age,sigma):
    price_times=np.asarray(price_times);history=np.asarray(history);age=np.asarray(age)
    j=np.searchsorted(price_times,t,side='right')-1
    n=history.shape[1];changes=[];valids=[];hs=[]
    if j<0:return np.zeros(n),np.zeros(n),np.zeros(n,dtype=int)
    for h in [6,24,72]:
        k=np.searchsorted(price_times,t-h*3600,side='right')-1
        if k<0:continue
        good=np.isfinite(history[j])&np.isfinite(history[k])&(history[j]>0)&(history[k]>0)&(age[j]<=3600)&(age[k]<=3600)
        r=np.log(np.divide(history[j],history[k],out=np.ones(n),where=good))
        changes.append(np.where(good,r,0.));valids.append(good);hs.append(h)
    if not hs:return np.zeros(n),np.zeros(n),np.zeros(n,dtype=int)
    r=np.stack(changes);good=np.stack(valids);h=np.asarray(hs)[:,None]
    count=good.sum(0);den=np.maximum(count,1)
    mu=(r*24/h).sum(0)/den
    strength=(np.tanh(r/np.sqrt(np.maximum(h*np.diag(sigma),1e-12)))*good).sum(0)/den
    return mu,strength,count

class VenueMomentum:
    def __init__(self,incumbent,activation,price_times,history,age):
        self.incumbent=incumbent;self.activation=activation;self.price_times=price_times
        self.history=history;self.age=age;self.last_decision=-1;self.trace=[]

    def propose(self,t,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        if t<self.activation:return self.incumbent.propose(t,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if kwargs.get('halted'):return dict(target_qty=np.zeros_like(qty).tolist(),risk_forced=True,defensive=True)
        if t%21600 or t<=self.last_decision:return None
        self.last_decision=t;old=qty*prices/equity;risk=protocol['risk']
        mu,strength,count=features(t,self.price_times,self.history,self.age,sigma)
        # Equal risk budget across the entire universe, signal-scaled; no
        # selected-asset concentration or leverage increase to win the replay.
        daily=np.sqrt(np.maximum(24*np.diag(sigma),1e-12))
        target=strength*risk['daily_volatility']/np.sqrt(len(qty))/daily
        cost=2*(protocol['execution']['fee_bps']+10)/10000+.0005
        ready=(count>=2)&(np.sign(target)*mu>cost)
        adding=(old*target<=0)|(abs(target)>abs(old)+1e-12)
        target=np.where(adding&~ready,np.where(old*target>0,old,0.),target)
        # Missing histories cannot manufacture a zero-target liquidation.
        target=np.where(count<2,old,target)
        target=project(target,equity,sigma,risk)
        turn=abs(target-old).sum()
        if turn>risk['normal_turnover_fraction']:target=old+(target-old)*risk['normal_turnover_fraction']/turn
        target=project(target,equity,sigma,risk);target=np.where(target*old<0,0.,target)
        units=np.array([rounded(v,n) for v,n in zip(target*equity/prices,protocol['execution']['decimals'])])
        assert not risk_breaches(units,equity,prices,sigma,risk)
        self.trace.append(dict(time=int(t),past_only_drift24h=mu.tolist(),strength=strength.tolist(),
            valid_horizons=count.tolist(),entry_cost_pass=ready.tolist(),target_qty=units.tolist()))
        return dict(target_qty=units.tolist(),risk_forced=False,defensive=False)
