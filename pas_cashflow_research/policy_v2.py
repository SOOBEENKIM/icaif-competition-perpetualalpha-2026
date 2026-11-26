"""Venue cash-flow allocation: past-only inputs, actual inventory and costs."""
import itertools
import numpy as np
from pas_economic_allocation.policy import solve,utility
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

class VenueController:
    def __init__(self,times,forecast,carry,fresh,mode,price_times=None,prices=None,impact_bps=10,horizon=72,receipt=.5):
        self.times=np.asarray(times);self.forecast=np.asarray(forecast);self.carry=np.asarray(carry)
        self.fresh=np.asarray(fresh);self.mode=mode;self.trace=[];self.last_decision=-1;self.impact_bps=impact_bps;self.horizon=horizon;self.receipt=receipt

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        if time<=self.last_decision or time%21600:return None
        i=int(np.searchsorted(self.times,time));assert i<len(self.times) and self.times[i]==time
        self.last_decision=int(time)
        old=qty*prices/equity;risk=protocol['risk'];f=self.carry[i]
        mu=self.forecast[i]-(1+self.receipt)/2*f # full payments, half receipts via extra absolute penalty
        cov=self.horizon*np.asarray(sigma);cost=(protocol['execution']['fee_bps']+self.impact_bps)/10000
        exit_cost=cost+.0005+(1-self.receipt)/2*abs(f)
        bounds=np.full(len(old),min(risk['asset_fraction'],risk['asset_notional']/equity)*risk['sizing_headroom'])
        pairs=[]
        if self.mode=='funding_pairs72':
            best=utility(project(old,equity,sigma,risk),old,mu,cov,5.,cost,exit_cost)
            target=project(old,equity,sigma,risk)
            for a,b in itertools.combinations(range(len(old)),2):
                if not(self.fresh[i,a] and self.fresh[i,b]):pairs.append(dict(ids=[a,b],available=False));continue
                # Minimum-variance positive hedge from past covariance; both
                # signs are compared. No preferred market or winner whitelist.
                beta=float(sigma[a,b]/max(sigma[b,b],1e-12))
                if beta<=0:pairs.append(dict(ids=[a,b],available=True,qualified=False));continue
                base=np.zeros(len(old));base[a]=1/(1+beta);base[b]=-beta/(1+beta)
                choices=[]
                for sign in [-1,1]:
                    w=project(sign*base*min(6000/equity,risk['gross_fraction']),equity,sigma,risk)
                    val=utility(w,old,mu,cov,5.,cost,exit_cost)
                    choices.append(val)
                    if val>best+1e-12:best=val;target=w
                pairs.append(dict(ids=[a,b],available=True,qualified=True,beta=beta,utility=choices))
        else:
            target=solve(old,mu,cov,5.,cost,exit_cost,bounds)
        # Missing/stale PAS observations may reduce existing risk but cannot
        # establish or enlarge a position. All symbols are assessed each time.
        adding=(target*old<0)|(abs(target)>abs(old)+1e-12)
        target=np.where(~self.fresh[i]&adding,old,target)
        target=project(target,equity,sigma,risk)
        turnover=abs(target-old).sum()
        if turnover>risk['normal_turnover_fraction']:
            target=old+(target-old)*risk['normal_turnover_fraction']/turnover
            target=project(target,equity,sigma,risk)
        # Reversals first flatten; a later scheduled decision may open the new side.
        target=np.where(target*old<0,0.,target)
        feasible=project(old,equity,sigma,risk)
        gain=utility(target,old,mu,cov,5.,cost,exit_cost)-utility(feasible,old,mu,cov,5.,cost,exit_cost)
        if gain<=1e-12:target=feasible
        q=np.array([rounded(v,n) for v,n in zip(target*equity/prices,protocol['execution']['decimals'])])
        assert not risk_breaches(q,equity,prices,sigma,risk)
        self.trace.append(dict(time=int(time),price_mu=self.forecast[i].tolist(),carry=f.tolist(),fresh=self.fresh[i].tolist(),
                               old=old.tolist(),target=(q*prices/equity).tolist(),expected_gain=max(0.,gain),pairs=pairs))
        return dict(target_qty=q.tolist(),risk_forced=False,defensive=False)
