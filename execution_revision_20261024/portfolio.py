"""Staged one-hour execution policy; live imports remain unchanged.

No forecast fitting, capital increase or resubmission of filled quantities.
Reductions after the entry deadline are clipped to current OWN base inventory.
They expire with the original six-hour signal and cannot reverse a position.
"""
import copy
import numpy as np
from candidate_handoff.portfolio import CandidatePortfolio, target_from_frame
from candidate_handoff.risk_priority import RiskPriorityPortfolio
from candidate_handoff.producer import validate
from candidate_handoff.entry_policy import BASE
from strategy_final_risk.candidate import publication
from replay_lab.common_comparison.engine import risk_breaches

REVISION = 'one_hour_intent_persistent_reductions_v1'
ENTRY_SECONDS = 3600

def reduction_target(held, desired):
    held=np.asarray(held,float);desired=np.asarray(desired,float)
    if held.shape!=desired.shape or not np.isfinite([held,desired]).all():
        raise ValueError('Invalid reduction target')
    same=held*desired>0
    return np.where(same,np.sign(held)*np.minimum(abs(held),abs(desired)),0.)

class ExecutionPortfolio(CandidatePortfolio):
    execution_revision=REVISION

    def step_candidate(self,now,streams,packet,profile):
        if packet is None:
            return RiskPriorityPortfolio.step(self,now,streams,None,[])
        body=validate(packet,profile,now,expected_policy=getattr(self,'execution_policy',BASE))
        f=body['frame'];old=body['original_signal']
        event=publication(old['publication'],f['anchor_weights'],np.array(f['protected'],bool))
        if self.base_task and now>=self.base_task.get('entry_expires',self.base_task['expires']):
            task=self.base_task
            if 'intent_source' not in task:
                # Old unsubmitted entry tasks are deliberately not revived.
                self.base_task=None
            elif now>=task['intent_source']+21600:
                self.base_task=None
            else:
                held=self.account.owners['base'].qty
                task['target']=reduction_target(held,task['target'])
                task['expires']=task['intent_source']+21600
                task['reductions_only']=True
                if not np.any(abs(task['target']-held)>1e-10):self.base_task=None
        normal=(now>=self.ready_at and not self.account.unknown and not self.halted
            and self.plan['paper_start_unix']<=now<self.plan['capture_end_unix']-self.plan['end_liquidation_seconds'])
        boundary=now//21600*21600
        # Preserve the established decision window: this revision extends
        # unfilled execution, not the age of a newly generated forecast.
        if normal and boundary>self.last_decision and now-boundary<=900 and f['source_time']==boundary:
            self.settlement(now*1000,self.account)
            marks,_=self.prices(now,streams)
            if marks is not None and not self.unpriced_flat_assets:
                equity=self.account.reconcile(marks)
                forced=bool(risk_breaches(self.account.physical.qty,equity,marks,np.asarray(f['sigma']),self.p['risk']))
                forced|=equity/max(self.peak,equity)-1<=-self.p['risk']['drawdown_halt']
                if not forced:
                    action,trace=target_from_frame(f,marks,self.account.owners['base'].qty,equity,self.p)
                    if action is not None:
                        self.base_task=dict(target=np.array(action['target_qty']),expires=boundary+ENTRY_SECONDS,
                            entry_expires=boundary+ENTRY_SECONDS,intent_source=boundary,reductions_only=False,
                            decision_prices=marks.copy())
                        self.last_decision=boundary
                        self.events.append(dict(time=now,source=f['source_time'],kind='budget_lifecycle_decision',trace=trace,
                            target_qty=action['target_qty'],signal_sha256=packet['sha256'],
                            core_revision=body.get('core_revision','legacy_core_v1'),execution_policy=body.get('execution_policy',BASE),
                            execution_revision=self.execution_revision,venue_history=f.get('venue_history'),universe_audit=body.get('universe_audit',[])))
        return RiskPriorityPortfolio.step(self,now,streams,event,old['forecasts'])

def adopt(portfolio):
    result=copy.deepcopy(portfolio);result.__class__=ExecutionPortfolio
    # A version switch preserves inventory and schedule, not old order intent.
    result.base_task=None
    for pair in [result.pair,*result.sessions]:
        if pair is not None:
            pair.account=result.account;pair.owner=result.account.owners['pair']
    return result
