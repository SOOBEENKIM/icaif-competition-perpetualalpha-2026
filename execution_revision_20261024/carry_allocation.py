"""PAS cash-flow allocation, independent of reference-market entry directions.

Only settled funding and completed earlier prices are read. No training labels,
symbol whitelist, wallet identities, or final-period rankings enter the rule.
This is an exposed-history research candidate, not installed in the live agent.
"""
import numpy as np
from pas_economic_allocation.policy import solve, utility
from replay_lab.common_comparison.engine import project, rounded, risk_breaches


class CarryAllocation:
    def __init__(self, times, carry, incumbent, activation, price_times, history, age, guarded=True):
        self.times=np.asarray(times);self.carry=np.asarray(carry);self.incumbent=incumbent
        self.activation=activation;self.price_times=np.asarray(price_times)
        self.history=np.asarray(history);self.age=np.asarray(age);self.guarded=guarded
        self.last_decision=-1;self.trace=[]

    def propose(self,t,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        if t<self.activation:
            return self.incumbent.propose(t,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if kwargs.get('halted'):
            return dict(target_qty=np.zeros_like(qty).tolist(),risk_forced=True,defensive=True)
        if t%21600 or t<=self.last_decision:return None
        i=int(np.searchsorted(self.times,t));assert self.times[i]==t
        self.last_decision=t;old=qty*prices/equity;risk=protocol['risk']
        carry=self.carry[i]
        # Forecast receipts at half face value, pay liabilities in full. The
        # absolute term plus linear term yields this asymmetry on either side.
        mu=-.75*carry;fee=protocol['execution']['fee_bps']/10000
        cost=fee+.001;exit_cost=cost+.0005+.25*abs(carry)
        cov=168*np.asarray(sigma);lam=5.
        bound=min(risk['asset_fraction'],risk['asset_notional']/equity)*risk['sizing_headroom']
        target=solve(old,mu,cov,lam,cost,exit_cost,np.full(len(old),bound))
        h=np.searchsorted(self.price_times,t,side='right')-1
        b=np.searchsorted(self.price_times,t-24*3600,side='right')-1
        fresh=np.zeros(len(old),bool);move=np.zeros(len(old));adverse=np.zeros(len(old),bool)
        if h>=0:
            fresh=(self.age[h]<=3600)&np.isfinite(self.history[h])&(self.history[h]>0)
        if self.guarded:
            if b<0:fresh[:]=False
            else:
                valid=fresh&(self.age[b]<=3600)&np.isfinite(self.history[b])&(self.history[b]>0)
                move=np.log(np.divide(self.history[h],self.history[b],out=np.ones(len(old)),where=valid))
                # A one-daily-sigma adverse price move vetoes carry additions.
                # It does not manufacture a forecast that price must revert.
                adverse=np.sign(target)*move < -np.sqrt(np.maximum(24*np.diag(sigma),1e-12))
                fresh&=valid
        adding=(old*target<=0)|(abs(target)>abs(old)+1e-12)
        target=np.where(adding&(~fresh|adverse),np.where(old*target>0,old,0.),target)
        target=project(target,equity,sigma,risk)
        turn=abs(target-old).sum()
        if turn>risk['normal_turnover_fraction']:target=old+(target-old)*risk['normal_turnover_fraction']/turn
        target=project(target,equity,sigma,risk)
        target=np.where(target*old<0,0.,target)
        units=np.array([rounded(v,n) for v,n in zip(target*equity/prices,protocol['execution']['decimals'])])
        assert not risk_breaches(units,equity,prices,sigma,risk)
        self.trace.append(dict(time=int(t),carry=carry.tolist(),adverse_price_veto=adverse.tolist(),
            past24h_return=move.tolist(),fresh=fresh.tolist(),target_qty=units.tolist(),
            utility_gain=utility(units*prices/equity,old,mu,cov,lam,cost,exit_cost)-utility(old,old,mu,cov,lam,cost,exit_cost)))
        return dict(target_qty=units.tolist(),risk_forced=False,defensive=False)
