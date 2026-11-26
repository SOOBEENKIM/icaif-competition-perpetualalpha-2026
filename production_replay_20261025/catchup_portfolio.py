"""Once per six-hour block, use the first available fresh source.

Research-only scheduling repair. The exact boundary source can be absent or
late. A later completed current-hour source may supply the same allocation
rule, with actual timestamps and current PAS prices; never backdate a trade.
No extra rebalance in a block already decided, no resurrected stale forecast.
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
from execution_revision_20261024.portfolio import reduction_target


class CatchupPortfolio(GuardedPortfolio):
    execution_revision='fresh_source_once_per_six_hour_block_v1'

    def step_candidate(self,now,streams,packet,profile):
        if packet is None:return RiskPriorityPortfolio.step(self,now,streams,None,[])
        body=validate(packet,profile,now,expected_policy=getattr(self,'execution_policy',BASE))
        f=body['frame'];old=body['original_signal'];source=f['source_time']
        event=publication(old['publication'],f['anchor_weights'],np.asarray(f['protected'],bool))
        if self.base_task and now>=self.base_task.get('entry_expires',self.base_task['expires']):
            task=self.base_task;end=task.get('block_end',task['expires'])
            if now>=end:self.base_task=None
            else:
                held=self.account.owners['base'].qty;task['target']=reduction_target(held,task['target'])
                task['expires']=end;task['reductions_only']=True
                if not np.any(abs(task['target']-held)>1e-10):self.base_task=None
        boundary=now//21600*21600
        normal=(now>=self.ready_at and not self.account.unknown and not self.halted
            and self.plan['paper_start_unix']<=now<self.plan['capture_end_unix']-self.plan['end_liquidation_seconds'])
        if normal and boundary>self.last_decision and boundary<=source<=now and now-source<3600:
            self.settlement(now*1000,self.account);marks,_=self.prices(now,streams)
            if marks is not None and not self.unpriced_flat_assets:
                equity=self.account.reconcile(marks);sigma=np.asarray(f['sigma'])
                forced=bool(risk_breaches(self.account.physical.qty,equity,marks,sigma,self.p['risk']))
                forced|=equity/max(self.peak,equity)-1<=-self.p['risk']['drawdown_halt']
                if not forced:
                    action,trace=target_from_frame(f,marks,self.account.owners['base'].qty,equity,self.p)
                    if action is not None:
                        expiry=min(now+3600,boundary+21600)
                        self.base_task=dict(target=np.asarray(action['target_qty']),expires=expiry,entry_expires=expiry,
                            block_end=boundary+21600,decision_prices=marks.copy(),intent_source=source,reductions_only=False)
                        self.last_decision=boundary
                        self.events.append(dict(time=now,source=source,block=boundary,kind='fresh_block_catchup_decision',
                            source_age=now-source,trace=trace,target_qty=action['target_qty'],signal_sha256=packet['sha256'],execution_revision=self.execution_revision))
        return RiskPriorityPortfolio.step(self,now,streams,event,old['forecasts'])


def adopt(portfolio):
    from competition_live.live_account import LiveAccount
    if type(portfolio.account) not in (LiveAccount,GuardedAccount):raise TypeError('Unknown account behavior')
    p=copy.deepcopy(portfolio);p.__class__=CatchupPortfolio;p.account.__class__=GuardedAccount
    p.account.base_entry_anchor=None;p.base_task=None
    for pair in [p.pair,*p.sessions]:
        if pair is not None:pair.account=p.account;pair.owner=p.account.owners['pair']
    return p
