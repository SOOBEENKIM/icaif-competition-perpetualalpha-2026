"""Freeze the executable side of the decision book, rather than its mark.

An execution hypothesis, not an alpha model or a claim that the existing mark
guard is an implementation error. The original 10bp slippage band, expiry,
economic limits, inventory ownership and portfolio risk caps remain in force.
"""
import copy
import numpy as np
from competition_live.live_account import LiveAccount
from experiments.shared_execution_pairs_20261017.account_v2 import difference
from candidate_handoff.portfolio import target_from_frame
from candidate_handoff.risk_priority import RiskPriorityPortfolio
from candidate_handoff.producer import validate
from candidate_handoff.entry_policy import BASE
from strategy_final_risk.candidate import publication
from replay_lab.common_comparison.engine import risk_breaches
from execution_revision_20261024.portfolio import ExecutionPortfolio, reduction_target


def freeze_sides(symbols, quotes):
    prices = np.full((len(symbols), 2), np.nan)
    for j, symbol in enumerate(symbols):
        quote = quotes.get(symbol)
        if quote is not None and all(quote['book']['levels']):
            bid, ask = (float(levels[0]['px']) for levels in quote['book']['levels'])
            if 0 < bid <= ask:
                prices[j] = bid, ask
    return prices


class QuoteAccount(LiveAccount):
    def seal(self, intent, p, sc):
        raw = copy.deepcopy(intent)
        if str(raw.get('packet_id', '')).startswith('base:'):
            old = self.owners['base'].qty
            limits = raw.setdefault('owner_limits', {}).setdefault('base', {})
            for j, new in enumerate(raw['owner_targets']['base']):
                delta = difference(new, old[j])
                if not delta or old[j]*delta < 0:
                    continue
                side = 1 if delta > 0 else -1
                anchor = getattr(self, 'frozen_side_prices', None)
                if anchor is None or np.shape(anchor) != (len(old), 2):
                    raise ValueError('New exposure requires frozen decision bid/ask')
                price = anchor[j, 1 if side > 0 else 0]
                if not np.isfinite(price) or price <= 0:
                    raise ValueError('Missing decision-side price')
                cap = float(price)*(1+side*.001)
                previous = limits.get(str(j))
                limits[str(j)] = cap if previous is None else min(cap,previous) if side>0 else max(cap,previous)
        return super().seal(raw, p, sc)


class QuotePortfolio(ExecutionPortfolio):
    execution_revision = 'one_hour_frozen_decision_side_quote_v1'

    def submit_baseline(self, now, target, streams, marks, sigma, forced):
        anchors = self.base_task.get('frozen_side_prices') if self.base_task else None
        self.account.frozen_side_prices = anchors
        # An asset missing a decision quote may still be reduced, but cannot
        # acquire a later, more favorable reference price under the old intent.
        held = self.account.owners['base'].qty
        adding = (held*target <= 0) | (abs(target) > abs(held))
        if not forced and np.any(adding):
            available = np.isfinite(anchors).all(1) if anchors is not None else np.zeros(len(held),bool)
            reduced = reduction_target(held,target)
            target = np.where(adding & ~available,reduced,target)
        try:
            return RiskPriorityPortfolio.submit_baseline(self,now,target,streams,marks,sigma,forced)
        finally:
            self.account.frozen_side_prices = None

    def step_candidate(self, now, streams, packet, profile):
        if packet is None:
            return RiskPriorityPortfolio.step(self,now,streams,None,[])
        body=validate(packet,profile,now,expected_policy=getattr(self,'execution_policy',BASE))
        f=body['frame'];old=body['original_signal']
        event=publication(old['publication'],f['anchor_weights'],np.array(f['protected'],bool))
        if self.base_task and now>=self.base_task.get('entry_expires',self.base_task['expires']):
            task=self.base_task
            if 'intent_source' not in task or now>=task['intent_source']+21600:
                self.base_task=None
            else:
                held=self.account.owners['base'].qty
                task['target']=reduction_target(held,task['target'])
                task['expires']=task['intent_source']+21600;task['reductions_only']=True
                if not np.any(abs(task['target']-held)>1e-10):self.base_task=None
        normal=(now>=self.ready_at and not self.account.unknown and not self.halted
            and self.plan['paper_start_unix']<=now<self.plan['capture_end_unix']-self.plan['end_liquidation_seconds'])
        boundary=now//21600*21600
        if normal and boundary>self.last_decision and now-boundary<=900 and f['source_time']==boundary:
            self.settlement(now*1000,self.account);marks,quotes=self.prices(now,streams)
            if marks is not None and not self.unpriced_flat_assets:
                equity=self.account.reconcile(marks);sigma=np.asarray(f['sigma'])
                forced=bool(risk_breaches(self.account.physical.qty,equity,marks,sigma,self.p['risk']))
                forced|=equity/max(self.peak,equity)-1<=-self.p['risk']['drawdown_halt']
                if not forced:
                    action,trace=target_from_frame(f,marks,self.account.owners['base'].qty,equity,self.p)
                    if action is not None:
                        anchors=freeze_sides(self.p['symbols'],quotes)
                        self.base_task=dict(target=np.array(action['target_qty']),expires=boundary+3600,
                            entry_expires=boundary+3600,intent_source=boundary,reductions_only=False,
                            decision_prices=marks.copy(),frozen_side_prices=anchors)
                        self.last_decision=boundary
                        self.events.append(dict(time=now,source=f['source_time'],kind='frozen_quote_decision',trace=trace,
                            target_qty=action['target_qty'],signal_sha256=packet['sha256'],
                            frozen_quotes=[[float(v) if np.isfinite(v) else None for v in row] for row in anchors],
                            execution_revision=self.execution_revision))
        return RiskPriorityPortfolio.step(self,now,streams,event,old['forecasts'])


def adopt(portfolio):
    result=copy.deepcopy(portfolio);result.__class__=QuotePortfolio
    result.account.__class__=QuoteAccount;result.account.frozen_side_prices=None
    result.base_task=None
    for pair in [result.pair,*result.sessions]:
        if pair is not None:pair.account=result.account;pair.owner=result.account.owners['pair']
    return result
