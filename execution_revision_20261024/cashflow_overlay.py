"""Causal PAS cash-flow overlay; no model fitting or market whitelist."""
import numpy as np
from pas_performance_research.soft_timing import SoftTimingController
from pas_economic_allocation.policy import solve,utility
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

def entry_eligible(time,target,price_times,history,age):
    votes=[];i=np.searchsorted(price_times,time,side='right')-1
    for hours in [3,24]:
        j=np.searchsorted(price_times,time-hours*3600,side='right')-1
        good=np.zeros(len(target),bool)
        if i>=0 and j>=0:
            a,b=history[i],history[j]
            good=np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)
            if age is not None:good&=(age[i]<=3600)&(age[j]<=3600)
            good&=np.sign(target)*np.log(np.divide(a,b,out=np.ones_like(a),where=good))<=0
        votes.append(good)
    return np.stack(votes).any(0)

class CashflowOverlay(SoftTimingController):
    def __init__(self,*args,venue_carry,activation=0,**kwargs):
        super().__init__(*args,**kwargs);self.venue_carry=np.asarray(venue_carry);self.activation=activation

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        action=super().propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if action is None or time<self.activation:return action
        i=int(np.searchsorted(self.times,time));original=np.asarray(action['target_qty']);base=original*prices/equity;old=qty*prices/equity
        carry=self.venue_carry[i];mu=self.forecast[i]-.75*carry;cov=168*sigma
        risk=protocol['risk'];bound=min(risk['asset_fraction'],risk['asset_notional']/equity)*risk['sizing_headroom']
        # Include the configured stressed fee in the forecast cost threshold.
        cost=(protocol['execution']['fee_bps']+10.)/10000;exit_cost=cost+.0005+.25*abs(carry)
        w=solve(old,mu,cov,self.lam,cost,exit_cost,np.full(len(qty),bound))
        # Preserve every core risk restriction and paid reduction. Add only
        # on the core's own side, or into an unoccupied core slot; never force
        # a short quota or reinflate a risk-reduced anchor.
        gate=np.asarray(self.protected[i]);core=self.anchors[i]
        add=(~gate)&(volume>0)&(trades>0)&((core==0)|(w*core>0))&(abs(w)>abs(base)+1e-12)
        add&=(abs(base)>=abs(old)-1e-12)|(old*base<0)
        # A previously zero core target has no side. Evaluate the NEW
        # candidate side rather than accidentally treating sign(0) as a pass.
        add&=entry_eligible(time,w,self.price_times,self.history,self.age)
        if not add.any():
            self.trace[-1].update(venue_cashflow_carry=carry.tolist(),cashflow_utility_gain=0.,cashflow_added_symbols=[])
            return action
        target=np.where(add,w,base)
        target=project(target,equity,sigma,risk)
        turnover=abs(target-old).sum()
        if turnover>risk['normal_turnover_fraction']:
            target=old+(target-old)*risk['normal_turnover_fraction']/turnover
            target=project(target,equity,sigma,risk)
        # Projection cannot relax a protected model cap or reverse an owner.
        target[gate]=base[gate]
        target=np.where(target*old<0,0.,target)
        q=np.array([rounded(v,n) for v,n in zip(target*equity/prices,protocol['execution']['decimals'])])
        candidate=q*prices/equity
        gain=utility(candidate,old,mu,cov,self.lam,cost,exit_cost)-utility(base,old,mu,cov,self.lam,cost,exit_cost)
        if gain<=1e-12 or not np.any(abs(q)>abs(original)+1e-10) or risk_breaches(q,equity,prices,sigma,risk):q=original.copy()
        action['target_qty']=q.tolist()
        self.trace[-1].update(venue_cashflow_carry=carry.tolist(),cashflow_utility_gain=float(gain),cashflow_added_symbols=np.flatnonzero(abs(q)>abs(original)+1e-10).tolist())
        return action
