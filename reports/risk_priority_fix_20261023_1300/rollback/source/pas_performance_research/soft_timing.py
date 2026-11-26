"""Entry filters act on additional inventory, never erase a held thesis."""
import numpy as np
from pas_economic_allocation.lifecycle import LifecycleController
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

class SoftTimingController(LifecycleController):
    def __init__(self,*args,price_times,prices,entry_rule,age=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.price_times=np.asarray(price_times);self.history=np.asarray(prices)
        self.entry_rule=entry_rule;self.age=age

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        action=super().propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if action is None:return None
        target=np.asarray(action['target_qty']);original=target.copy()
        horizons=[3,24] if self.entry_rule=='pullback_any3' else [1,24] if self.entry_rule=='pullback_any1' else [6] if self.entry_rule=='pullback6_soft' else [6,24]
        votes=[]
        for horizon in horizons:
            i=np.searchsorted(self.price_times,time,side='right')-1
            j=np.searchsorted(self.price_times,time-horizon*3600,side='right')-1
            good=np.zeros(len(prices),bool);change=np.zeros(len(prices))
            if i>=0 and j>=0:
                a=self.history[i];b=self.history[j];good=np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)
                if self.age is not None:good&=(self.age[i]<=3600)&(self.age[j]<=3600)
                change=np.log(np.where(good,a/b,1.))
            votes.append(good & (np.sign(target)*change<=0))
        votes=np.stack(votes)
        ok=votes.any(0)
        residual=.25 if self.entry_rule.endswith('soft') else 0.
        sign=np.sign(target)
        if self.entry_rule=='entry_carry':
            k=np.searchsorted(self.times,time);edge=sign*self.forecast[k]-np.where(sign*self.carry[k]>0,sign*self.carry[k],.5*sign*self.carry[k])
            ok=edge>.0043
        for k in range(len(target)):
            if ok[k]:continue
            if target[k]*qty[k]<=0:target[k]*=residual # Close old opposite side, don't reopen.
            elif abs(target[k])>abs(qty[k]):target[k]=qty[k]+residual*(target[k]-qty[k]) # Hold existing same-side units.
            # A base exit, reduction or hard-risk trim always survives.
        weights=project(target*prices/equity,equity,sigma,protocol['risk'])
        q=np.array([rounded(v,n) for v,n in zip(weights*equity/prices,protocol['execution']['decimals'])])
        if risk_breaches(q,equity,prices,sigma,protocol['risk']):raise AssertionError('Entry filter risk violation')
        if np.any(abs(q)>abs(original)+1e-10):raise AssertionError('Entry filter restored reduced position')
        action['target_qty']=q.tolist();self.trace[-1].update(entry_filter_pass=ok.tolist(),pre_entry_filter_qty=original.tolist(),target_qty=q.tolist())
        return action
