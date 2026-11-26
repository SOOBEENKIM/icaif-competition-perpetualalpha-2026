"""Cost-aware signed allocation from actual inventory, with hard final caps."""
import numpy as np
from replay_lab.common_comparison.engine import project,rounded,risk_breaches


def utility(w,old,mu,cov,lam,cost,exit_cost):
    return float(mu@w-.5*lam*(w@cov@w)-cost*abs(w-old).sum()-np.sum(exit_cost*abs(w)))


def solve(old,mu,cov,lam,cost,exit_cost,bound):
    """Exact coordinate minimizers of the box-constrained convex objective.

    Global portfolio caps are applied once by the controller. We check utility
    after that projection; no claim of globally optimal constrained allocation.
    """
    w=np.clip(old,-bound,bound).copy();exit_cost=np.broadcast_to(exit_cost,old.shape)
    for _ in range(100):
        before=w.copy()
        for j in range(len(w)):
            a=max(lam*cov[j,j],1e-12)
            b=mu[j]-lam*(cov[j]@w-cov[j,j]*w[j])
            points=[-bound[j],bound[j],0.,float(np.clip(old[j],-bound[j],bound[j]))]
            breaks=sorted(set([-bound[j],bound[j],0.,float(np.clip(old[j],-bound[j],bound[j]))]))
            for lo,hi in zip(breaks[:-1],breaks[1:]):
                mid=(lo+hi)/2
                root=(b-cost*np.sign(mid-old[j])-exit_cost[j]*np.sign(mid))/a
                points.append(float(np.clip(root,lo,hi)))
            values=[b*v-.5*a*v*v-cost*abs(v-old[j])-exit_cost[j]*abs(v) for v in points]
            w[j]=points[int(np.argmax(values))]
        if abs(w-before).max()<1e-10:break
    return w


class EconomicController:
    def __init__(self,times,forecast,carry,lam,mode='unified',horizon=168):
        self.times=np.asarray(times);self.forecast=np.asarray(forecast);self.carry=np.asarray(carry)
        self.lam=float(lam);self.mode=mode;self.horizon=horizon;self.last_decision=-1;self.trace=[]

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,buffer=.2,**kwargs):
        if time<=self.last_decision or time%21600:return None
        i=int(np.searchsorted(self.times,time))
        if i>=len(self.times) or self.times[i]!=time:raise ValueError('Unaligned decision')
        if not np.isfinite(prices).all() or np.any(prices<=0) or equity<=0:raise ValueError('Bad account input')
        old=np.asarray(qty)*prices/equity
        price_mu=self.forecast[i] if self.mode!='carry_only' else np.zeros_like(old)
        carry=self.carry[i]
        # -0.75*f*w -0.25*abs(f*w): full forecast payments, half receipts.
        mu=np.nan_to_num(price_mu,nan=0.)-.75*carry
        missing=~np.isfinite(price_mu) if self.mode!='carry_only' else np.zeros_like(old,dtype=bool)
        cost=.0019;exit_cost=.0019+.0005+.25*abs(carry)
        risk=protocol['risk'];cap=min(risk['asset_fraction'],risk['asset_notional']/equity)*risk['sizing_headroom']
        bounds=np.full(len(old),cap)
        cov=self.horizon*np.asarray(sigma)
        target=solve(old,mu,cov,self.lam,cost,exit_cost,bounds)
        # Missing price forecasts do not permit opening a pure price position.
        # Known carry remains usable on all assets; fresh market data mandatory.
        fresh=(volume>0)&(trades>0)
        target=np.where(~fresh & ((target*old<0)|(abs(target)>abs(old))),old,target)
        target=project(target,equity,sigma,risk)
        feasible_old=project(old,equity,sigma,risk)
        gain=utility(target,old,mu,cov,self.lam,cost,exit_cost)-utility(feasible_old,old,mu,cov,self.lam,cost,exit_cost)
        if gain<=1e-12:target=feasible_old
        # A single net target; no later volatility rescaling may increase it.
        q=np.array([rounded(v,d) for v,d in zip(target*equity/prices,protocol['execution']['decimals'])])
        if risk_breaches(q,equity,prices,sigma,risk):raise AssertionError('Final risk cap violated')
        self.last_decision=int(time)
        self.trace.append(dict(time=int(time),price_mu=price_mu.tolist(),carry=carry.tolist(),net_mu=mu.tolist(),
            old=old.tolist(),target=(q*prices/equity).tolist(),expected_utility_gain=max(gain,0.),
            gross_budget=protocol['initial'],unpriced=missing.tolist()))
        return dict(target_qty=q.tolist(),risk_forced=False,defensive=False)
