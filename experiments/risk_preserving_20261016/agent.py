"""Standalone candidate agent. Accepts history/actual positions, returns intents."""
import numpy as np
from strategy import load_agent
from .policy import target_path,RiskCapController


class RiskPreservingAgent:
    PROFILE='final_risk_cap_v1'

    def __init__(self,saved=None):
        if saved is not None and (saved.get('schema')!=1 or saved.get('profile')!=self.PROFILE):
            raise ValueError('Cannot reuse incumbent/different candidate execution state')
        state=saved or {}
        self.incumbent=load_agent('default')
        self.controller=RiskCapController(state.get('control'))
        self.peak=float(state.get('peak',1e6));self.halted=bool(state.get('halted',False))
        self.last_observation=int(state.get('last_observation',-1));self.market=None

    def prepare_history(self,market,original,*,funding_source):
        self.market=None
        self.incumbent.prepare_history(market,original,funding_source=funding_source)
        a=self.incumbent.agent
        self.components=target_path(market,self.incumbent.weights,a.base_weights,a.frozen_probability,a.signal.forecast)
        trend=a.frozen_baseline_weights*(a.selection['risk_scale']*a.extra_scale)
        self.components.update(scaled_trend_before_blend=trend,
            reversion_and_blend_cap_delta=self.incumbent.weights-trend,
            final_risk_adjustment=self.components['candidate']-self.incumbent.weights)
        np.testing.assert_allclose(trend+self.components['reversion_and_blend_cap_delta']+
            self.components['final_risk_adjustment'],self.components['candidate'],atol=1e-15,rtol=0)
        self.weights=self.components['candidate'];self.retention=self.components['retention']
        self.sigma=a.sigma;self.protocol=a.bundle['protocol'];self.market=market

    def decide(self,time,qty,equity,valid_input=True):
        if (self.market is None or not valid_input or not np.isfinite(time) or time!=int(time)
            or time<=max(self.last_observation,self.controller.last_decision)
            or time<self.incumbent.agent.deployment_start or not np.isfinite(equity) or equity<=0):return None
        qty=np.asarray(qty,dtype=float)
        if qty.shape!=(10,) or not np.isfinite(qty).all():return None
        try:i=self.market.exact(time)
        except ValueError:return None
        if not np.isfinite(self.sigma[i]).all():return None
        self.peak=max(self.peak,equity);self.halted|=equity/self.peak-1<=-self.protocol['risk']['drawdown_halt']
        self.last_observation=int(time)
        action=self.controller.propose(time,self.market.close[i],qty,equity,self.weights[i],self.sigma[i],
            self.market.volume[i],self.market.trades[i],self.protocol,buffer=.2,halted=self.halted,retention=self.retention[i])
        if action is not None:
            action['signal_attribution']={k:self.components[k][i].tolist() for k in
                ('incumbent','underlying_trend','candidate','retention','regime_retention','chronos_retention',
                 'forecast','frozen_probability','scaled_trend_before_blend','reversion_and_blend_cap_delta','final_risk_adjustment')}
        return action

    def snapshot(self):
        return {'schema':1,'profile':self.PROFILE,'control':self.controller.snapshot(),'peak':self.peak,
                'halted':bool(self.halted),'last_observation':self.last_observation}
