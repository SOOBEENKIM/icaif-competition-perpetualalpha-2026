"""Use the exact selected base policy; retain the live pair state machine."""
import copy
import numpy as np
from pas_economic_allocation.lifecycle import LifecycleController
from strategy_final_risk.candidate import FinalRiskPortfolio,publication
from replay_lab.common_comparison.engine import risk_breaches
from .producer import validate


def target_from_frame(frame,prices,qty,equity,protocol):
    # Clock zero is only a one-row policy evaluation, not a market timestamp.
    # Actual cadence/availability is checked by step(). This also permits a
    # clearly labelled "if rebalance were now" unsigned preview between rounds.
    c=LifecycleController([0],np.array([frame['price_forecast_168h']]),
        np.array([frame['reference_funding_forecast_168h']]),np.array([frame['anchor_weights']]),
        np.array([frame['protected']],bool),frame['risk_aversion'])
    a=c.propose(0,np.asarray(prices),np.asarray(qty),equity,np.asarray(frame['anchor_weights']),
        np.asarray(frame['sigma']),np.asarray(frame['volume']),np.asarray(frame['trades']),protocol,buffer=.2)
    return a,c.trace[-1] if c.trace else None


class CandidatePortfolio(FinalRiskPortfolio):
    def step_candidate(self,now,streams,packet,profile):
        if packet is None:return super().step(now,streams,None,[])
        body=validate(packet,profile,now);f=body['frame'];old=body['original_signal']
        event=publication(old['publication'],f['anchor_weights'],np.array(f['protected'],bool))
        # Heavy inference is already complete. All ensuing sizes use current
        # PAS marks and actual acquired base inventory, including pre-switch fills.
        normal=(now>=self.ready_at and not self.account.unknown and not self.halted
            and self.plan['paper_start_unix']<=now<self.plan['capture_end_unix']-self.plan['end_liquidation_seconds'])
        boundary=now//21600*21600
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
                        self.base_task=dict(target=np.array(action['target_qty']),expires=boundary+900)
                        self.last_decision=boundary
                        self.events.append(dict(time=now,source=f['source_time'],kind='budget_lifecycle_decision',trace=trace,
                            target_qty=action['target_qty'],signal_sha256=packet['sha256'],
                            core_revision=body.get('core_revision','legacy_core_v1'),
                            universe_audit=body.get('universe_audit',[])))
        # Parent skips recalculation because last_decision was set above, but
        # executes this exact task and retains pair entry/repair/exit behavior.
        return super().step(now,streams,event,old['forecasts'])


def adopt(portfolio):
    # Preserve positions, entries, realized PnL, peak, halt, order identities,
    # pair lifecycle and scheduling. Only discard old unsubmitted base intent.
    result=copy.deepcopy(portfolio);result.__class__=CandidatePortfolio
    result.base_task=None
    for pair in [result.pair,*result.sessions]:
        if pair is not None:
            pair.account=result.account;pair.owner=result.account.owners['pair']
    return result
