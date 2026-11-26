"""Offline sleeve ownership: partial fills are never treated as full targets."""
import numpy as np
from pas_performance_research.soft_timing import SoftTimingController
from .policy import allocate
from .market import market_inputs


class Sleeves:
    def __init__(self,n):
        self.base=np.zeros(n);self.extra=np.zeros(n);self.base_target=np.zeros(n)
        self.extra_target=np.zeros(n);self.last_intent=0;self.opened=np.full(n,-1,dtype=np.int64)

    def reconcile(self,physical):
        physical=np.asarray(physical,float)
        for j,new in enumerate(physical):
            before=self.base[j]+self.extra[j]
            if new==before:continue
            if before*new<0:
                # Reduction-before-reversal is enforced by the exchange simulator.
                self.base[j]=new;self.extra[j]=0.;self.opened[j]=-1;continue
            if abs(new)<abs(before):
                removed=abs(before)-abs(new);take=min(removed,abs(self.extra[j]))
                self.extra[j]-=np.sign(before)*take
                self.base[j]=new-self.extra[j]
            else:
                added=abs(new)-abs(before);direction=np.sign(new)
                need=max(0.,abs(self.base_target[j])-abs(self.base[j])) if direction*self.base_target[j]>0 else 0.
                base_fill=min(added,need);extra_fill=added-base_fill
                self.base[j]+=direction*base_fill;self.extra[j]+=direction*extra_fill
                if abs(self.extra[j])>1e-10 and self.opened[j]<0:self.opened[j]=self.last_intent
            if abs(self.extra[j])<1e-10:self.extra[j]=0.;self.opened[j]=-1
        np.testing.assert_allclose(self.base+self.extra,physical,rtol=0,atol=1e-9)


class OpportunityController:
    def __init__(self,baseline,reference,reference_forecast,reference_uncertainty,data,funding,*,uniform=False):
        self.base=baseline;self.reference=reference;self.mu=reference_forecast;self.se=reference_uncertainty
        self.data=data;self.funding=funding;self.uniform=uniform;self.sleeves=Sleeves(len(funding));self.trace=[]

    def propose(self,at,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        if at%21600:return None
        self.sleeves.reconcile(qty)
        action=self.base.propose(at,prices,self.sleeves.base.copy(),equity,raw,sigma,volume,trades,protocol,**kwargs)
        if action is None:return None
        base=np.asarray(action['target_qty']);trace=self.base.trace[-1]
        i=int(np.searchsorted(self.base.times,at));ri=int(np.searchsorted(self.reference['market_times'],at))
        if ri>=len(self.reference['market_times']) or self.reference['market_times'][ri]!=at:raise ValueError('Unaligned forecast')
        info=market_inputs(at,prices,self.data,self.reference,self.funding)
        expected=np.expm1(np.clip(self.mu[ri]+info['basis_log_change'],-.5,.5))
        signed_funding=np.sign(base)*info['funding_24h']
        # Signed expected return, with full payments and half uncertain receipts.
        expected-=np.sign(base)*np.where(signed_funding>0,signed_funding,.5*signed_funding)
        uncertainty=self.se[ri]+info['basis_uncertainty']
        reducing=(self.sleeves.base!=0)&((self.sleeves.base*base<=0)|(abs(base)<abs(self.sleeves.base)-1e-10))
        expired=(self.sleeves.opened>=0)&(at-self.sleeves.opened>=24*3600)
        safe=(~self.base.protected[i])&~reducing&info['fresh']&info['funding_ready']&~expired
        if action.get('risk_forced') or action.get('defensive') or kwargs.get('halted'):
            safe[:]=False
        allowed=safe&np.array(trace['entry_filter_pass'],bool)
        result=allocate(base,self.sleeves.extra,prices,equity,sigma,protocol['risk'],protocol['execution']['decimals'],
                        expected,uncertainty,info['entry_cost'],info['exit_cost'],info['capacity_notional'],
                        allowed,safe,uniform=self.uniform)
        self.sleeves.base_target=base.copy();self.sleeves.extra_target=np.asarray(result['extra_qty'])
        self.sleeves.last_intent=at
        action['target_qty']=result['target_qty']
        self.trace.append(dict(time=int(at),base_qty=base.tolist(),base_actual=self.sleeves.base.tolist(),
            extra_actual=self.sleeves.extra.tolist(),expired=expired.tolist(),protected=self.base.protected[i].tolist(),
            forecast=expected.tolist(),uncertainty=uncertainty.tolist(),allowed=allowed.tolist(),
            market={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in info.items()},**result))
        return action
