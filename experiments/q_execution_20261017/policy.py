"""Actual-holding based adjustment. No forecasts, clocks or risk caps changed."""
import numpy as np
from experiments.risk_preserving_20261016.policy import RiskCapController
from replay_lab.common_comparison.engine import project, rounded, risk_breaches


class AdjustmentController(RiskCapController):
    def __init__(self, policy, saved=None):
        if policy not in ('point','partial_6h','partial_24h','band_edge'):
            raise ValueError('Unknown execution policy')
        if saved is not None and saved.get('adjustment_policy')!=policy:
            raise ValueError('Cannot restore another policy state')
        super().__init__(saved)
        self.policy=policy

    def snapshot(self):
        return {**super().snapshot(),'adjustment_policy':self.policy}

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        action=super().propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if action is None or self.policy=='point' or action['risk_forced']:
            return action
        target=np.array(action['target_qty'],float);sealed=target.copy();qty=np.asarray(qty,float)
        same=target*qty>0
        opening=(qty==0)&(target!=0)
        # Flat/reversal decisions preserve immediate liquidation. Ordinary
        # same-direction adjustments may be staged, but hard caps never are.
        if self.policy.startswith('partial'):
            target=np.where(same|opening,qty+.5*(target-qty),target)
        elif self.policy=='band_edge':
            low=.8*abs(target);high=abs(target)/.8
            nearest=np.clip(abs(qty),low,high)*np.sign(target)
            target=np.where(same|opening,nearest,target)
        if self.policy=='partial_24h' and time%86400!=43200:
            # Decreases/flat exits remain on the inherited six-hour clock.
            # New or increased exposure waits for 12:00 UTC.
            grow=(target*qty<0)|(abs(target)>abs(qty))
            target=np.where(grow,np.where(target*qty<0,0.,qty),target)
        active=np.array(action['risk_cap_active'],bool);ceiling=np.array(action['risk_cap_qty'])
        target=np.where(active,np.sign(target)*np.minimum(abs(target),ceiling),target)
        target=project(target*prices/equity,equity,sigma,protocol['risk'])*equity/prices
        target=np.array([rounded(q,d) for q,d in zip(target,protocol['execution']['decimals'])])
        if risk_breaches(target,equity,prices,sigma,protocol['risk']):
            target=np.array([rounded(q*.999999,d) for q,d in zip(target,protocol['execution']['decimals'])])
            if risk_breaches(target,equity,prices,sigma,protocol['risk']):target=np.zeros_like(qty)
        if np.any(abs(target[active])>ceiling[active]+1e-10):raise AssertionError('Risk cap restored')
        # No future-fill assumption: subsequent calls use reconciled holdings,
        # not the previous order's requested quantity.
        action.update(target_qty=target.tolist(),adjustment_policy=self.policy,
            point_target_qty=sealed.tolist(),decision_reason='staged_adjustment')
        return action


class HistoricalController(AdjustmentController):
    def __init__(self,policy,times,retention):
        super().__init__(policy);self.times=times;self.retention=retention

    def propose(self,time,*args,**kwargs):
        i=int(np.searchsorted(self.times,time))
        if i>=len(self.times) or self.times[i]!=time:raise ValueError('Missing signal time')
        return super().propose(time,*args,retention=self.retention[i],**kwargs)
