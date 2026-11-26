"""Fresh hourly allocation; unchanged alpha, costs, sizing and risk ceilings.

Research candidate only. Each intent expires15min after its own source hour.
No resurrected old signal, no required transaction count, no wider price band.
"""
import copy
import numpy as np
from candidate_handoff.producer import validate
from candidate_handoff.entry_policy import BASE
from candidate_handoff.portfolio import target_from_frame
from candidate_handoff.risk_priority import RiskPriorityPortfolio
from strategy_final_risk.candidate import publication
from replay_lab.common_comparison.engine import risk_breaches
from execution_revision_20261024.guarded_portfolio import GuardedPortfolio,GuardedAccount

class HourlyPortfolio(GuardedPortfolio):
    execution_revision='fresh_hourly_15min_fixed_price_v1'

    def step_candidate(self,now,streams,packet,profile):
        if packet is None:return RiskPriorityPortfolio.step(self,now,streams,None,[])
        body=validate(packet,profile,now,expected_policy=getattr(self,'execution_policy',BASE))
        f=body['frame'];old=body['original_signal']
        event=publication(old['publication'],f['anchor_weights'],np.array(f['protected'],bool))
        boundary=now//3600*3600
        normal=(now>=self.ready_at and not self.account.unknown and not self.halted
            and self.plan['paper_start_unix']<=now<self.plan['capture_end_unix']-self.plan['end_liquidation_seconds'])
        if normal and boundary>self.last_decision and now-boundary<=900 and f['source_time']==boundary:
            self.settlement(now*1000,self.account);marks,_=self.prices(now,streams)
            if marks is not None and not self.unpriced_flat_assets:
                equity=self.account.reconcile(marks);sigma=np.asarray(f['sigma'])
                forced=bool(risk_breaches(self.account.physical.qty,equity,marks,sigma,self.p['risk']))
                forced|=equity/max(self.peak,equity)-1<=-self.p['risk']['drawdown_halt']
                if not forced:
                    action,trace=target_from_frame(f,marks,self.account.owners['base'].qty,equity,self.p)
                    if action is not None:
                        self.base_task=dict(target=np.asarray(action['target_qty']),expires=boundary+900,
                            decision_prices=marks.copy())
                        self.last_decision=boundary
                        self.events.append(dict(time=now,source=f['source_time'],kind='hourly_fresh_decision',trace=trace,
                            target_qty=action['target_qty'],signal_sha256=packet['sha256'],
                            execution_revision=self.execution_revision))
        # Parent model-risk path still has priority over additions. The original
        # coordinator cannot generate an old6h target: its boundary <= ours.
        return RiskPriorityPortfolio.step(self,now,streams,event,old['forecasts'])

def adopt(portfolio):
    from competition_live.live_account import LiveAccount
    if type(portfolio.account) not in (LiveAccount,GuardedAccount):raise TypeError('Unknown account behavior')
    p=copy.deepcopy(portfolio);p.__class__=HourlyPortfolio;p.account.__class__=GuardedAccount
    p.account.base_entry_anchor=None;p.base_task=None
    for pair in [p.pair,*p.sessions]:
        if pair is not None:pair.account=p.account;pair.owner=p.account.owners['pair']
    return p
