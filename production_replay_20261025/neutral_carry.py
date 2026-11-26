"""All-45-pair funding allocation; maturity-causal, no asset identities in rule.

This is carry, not a claim of cointegration or guaranteed price reversion.
One common long/short target under unchanged account caps. Targets are dollar
neutral; sequential partial fills can create bounded temporary net exposure.
"""
import numpy as np
from pas_economic_allocation.policy import utility
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

def solve_neutral(old,mu,cov,lam,cost,exit_cost,bound):
    """Pair-coordinate exact line searches for a convex neutral allocation.

    Starts at cash (always feasible); each direction adds j and subtracts k.
    Every pair considered, no configured pair list. Tie order deterministic.
    """
    n=len(old);w=np.zeros(n);exit_cost=np.broadcast_to(exit_cost,(n,));checks=0
    for iteration in range(200):
        largest=0.
        for j in range(n):
            for k in range(j+1,n):
                checks+=1
                lo=max(-bound[j]-w[j],w[k]-bound[k]);hi=min(bound[j]-w[j],w[k]+bound[k])
                if hi<=lo:continue
                a=max(lam*(cov[j,j]+cov[k,k]-2*cov[j,k]),1e-12)
                b=mu[j]-mu[k]-lam*((cov[j]-cov[k])@w)
                knots=sorted(set([lo,hi]+[float(np.clip(x,lo,hi)) for x in [-w[j],w[k],old[j]-w[j],w[k]-old[k]]]))
                candidates=knots+[0.]
                for l,h in zip(knots[:-1],knots[1:]):
                    mid=(l+h)/2
                    slope=cost*(np.sign(w[j]+mid-old[j])-np.sign(w[k]-mid-old[k]))
                    slope+=exit_cost[j]*np.sign(w[j]+mid)-exit_cost[k]*np.sign(w[k]-mid)
                    candidates.append(float(np.clip((b-slope)/a,l,h)))
                def gain(delta):
                    return b*delta-.5*a*delta*delta-cost*(abs(w[j]+delta-old[j])+abs(w[k]-delta-old[k])-abs(w[j]-old[j])-abs(w[k]-old[k]))-exit_cost[j]*(abs(w[j]+delta)-abs(w[j]))-exit_cost[k]*(abs(w[k]-delta)-abs(w[k]))
                values=[gain(x) for x in candidates];ix=int(np.argmax(values));delta=candidates[ix]
                if values[ix]>1e-14:w[j]+=delta;w[k]-=delta;largest=max(largest,abs(delta))
        if largest<1e-9:break
    assert abs(w.sum())<1e-9 and np.all(abs(w)<=bound+1e-9)
    return w,dict(pair_directions_considered=n*(n-1)//2,coordinate_checks=checks,iterations=iteration+1)

class NeutralCarry:
    def __init__(self,times,carry,incumbent,activation,price_times,history,age):
        self.times=np.asarray(times);self.carry=np.asarray(carry);self.forecast=np.zeros_like(carry);self.receipt=.5
        self.incumbent=incumbent;self.activation=activation;self.price_times=price_times;self.history=history;self.age=age
        self.last_decision=-1;self.trace=[]

    def propose(self,t,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        if t<self.activation:return self.incumbent.propose(t,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if kwargs.get('halted'):return dict(target_qty=np.zeros_like(qty),risk_forced=True,defensive=True)
        if t%21600 or t<=self.last_decision:return None
        self.last_decision=t;i=int(np.searchsorted(self.times,t));assert self.times[i]==t
        old=qty*prices/equity;carry=self.carry[i];mu=-.75*carry
        cost=.0019;exit_cost=.0024+.25*abs(carry);cov=168*np.asarray(sigma);lam=5.
        risk=protocol['risk'];cap=min(risk['asset_fraction'],risk['asset_notional']/equity)*risk['sizing_headroom']
        k=np.searchsorted(self.price_times,t,side='right')-1
        fresh=(self.age[k]<=3600)&np.isfinite(self.history[k])&(self.history[k]>0)
        bounds=np.where(fresh,cap,0.)
        target,audit=solve_neutral(old,mu,cov,lam,cost,exit_cost,bounds)
        target=project(target,equity,sigma,risk)
        before_turn=target.copy();turn=abs(target-old).sum()
        if turn>risk['normal_turnover_fraction']:target=old+(target-old)*risk['normal_turnover_fraction']/turn
        target=project(target,equity,sigma,risk)
        # Reversals close the old position first; never simultaneous sides.
        target=np.where(target*old<0,0.,target)
        q=np.array([rounded(v,n) for v,n in zip(target*equity/prices,protocol['execution']['decimals'])])
        assert not risk_breaches(q,equity,prices,sigma,risk)
        self.trace.append(dict(time=int(t),carry=carry.tolist(),neutral_target=before_turn.tolist(),
            final_target=(q*prices/equity).tolist(),old=old.tolist(),fresh=fresh.tolist(),
            expected_utility=utility(q*prices/equity,old,mu,cov,lam,cost,exit_cost),**audit))
        return dict(target_qty=q.tolist(),risk_forced=False,defensive=False)
