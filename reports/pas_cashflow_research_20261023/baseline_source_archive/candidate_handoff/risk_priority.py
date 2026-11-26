"""Keep a valid rebalance intent across model-risk reductions.

The frozen coordinator still owns pairs, global stops, packets and actual fills.
Only model-cap scheduling changes; no forecast or training parameters change.
"""
import copy,time
import numpy as np
from competition_live.coordinator import Portfolio as Coordinator
from strategy_final_risk.candidate import FinalRiskPortfolio,validate
from replay_lab.common_comparison.engine import risk_breaches

REVISION='risk_priority_resume_v1'

class DeadlineGateway:
    """Reject locally before transmission when a normal task expires during I/O."""
    def __init__(self,gateway,deadline):self.gateway=gateway;self.deadline=deadline
    def __getattr__(self,name):return getattr(self.gateway,name)
    def preflight(self,plan,book):
        from joint_wallet.book import ReconciliationError
        if time.time()>=self.deadline:raise ReconciliationError('Rebalance intent expired before preflight')
        result=self.gateway.preflight(plan,book)
        if time.time()>=self.deadline:raise ReconciliationError('Rebalance intent expired during preflight')
        return result
    def send(self,plan):
        from bot.execution import OrderRejected
        if time.time()>=self.deadline:raise OrderRejected('Rebalance intent expired before transmission')
        return self.gateway.send(plan)


class RiskPriorityPortfolio(FinalRiskPortfolio):
    execution_revision=REVISION

    def defer_dust(self,target,marks,quotes,source):
        """Defer only proved subminimum reductions, never pretend they filled.

The live gateway repeats exact lot/price, depth, margin and team-net checks.
Missing books cannot prove dust. Full-close exceptions require BOTH the sleeve
adapter and the actual shared account to be fully closed by this delta.
"""
        held=self.account.owners['base'].qty
        result=target.copy();deferred=[]
        own=self.account.physical.qty
        reserved=getattr(self.account,'reserved',None)
        net=own+(reserved.qty if reserved is not None else 0.)
        for j,symbol in enumerate(self.p['symbols']):
            delta=float(target[j]-held[j])
            if abs(delta)<1e-10 or held[j]*delta>=0 or abs(delta)>abs(held[j])+1e-10:continue
            # Preserve internal base/pair transfers; they have no exchange minimum.
            if abs(self.account.owners['pair'].qty[j])>1e-10:continue
            quote=quotes.get(symbol) if quotes else None
            if quote is None:continue
            side=1 if delta>0 else -1
            best=float(quote['book']['levels'][1 if side>0 else 0][0]['px'])
            upper_price=max(float(marks[j]),best)*1.001
            if abs(delta)*upper_price>=10.:continue
            full_close=(own[j]*delta<0 and abs(delta+own[j])<1e-12
                        and net[j]*delta<0 and abs(delta+net[j])<1e-12)
            if full_close:continue
            result[j]=held[j]
            key=(source,symbol,float(held[j]),float(target[j]))
            seen=getattr(self,'risk_dust_seen',set())
            if key not in seen:
                seen.add(key);self.risk_dust_seen=seen
                deferred.append(dict(symbol=symbol,held_qty=float(held[j]),requested_qty=float(target[j]),
                    delta_notional_upper=abs(delta)*upper_price,reason='below_exchange_minimum_partial_reduction'))
        if deferred:self.events.append(dict(time=self._risk_step_time,kind='risk_cap_dust_deferred',source=source,items=deferred))
        return result

    def submit_baseline(self,now,target,streams,marks,sigma,forced):
        self._risk_step_time=now
        target,cut=self.protected_target(target,marks,self.account.reconcile(marks))
        if not forced:
            _,quotes=self.prices(now,streams)
            frame=getattr(self,'repair_frame',None)
            source=frame['source_time'] if frame else None
            target=self.defer_dust(target,marks,quotes,source)
        if not np.any(abs(target-self.account.owners['base'].qty)>1e-10):return
        executor=getattr(self.account,'executor',None)
        gateway=getattr(executor,'gateway',None)
        bounded=not forced and self.base_task is not None and gateway is not None
        if bounded:executor.gateway=DeadlineGateway(gateway,self.base_task['expires'])
        try:return Coordinator.submit_baseline(self,now,target,streams,marks,sigma,forced or cut)
        finally:
            if bounded:executor.gateway=gateway

    def step(self,now,streams,event,forecasts):
        self._risk_step_time=now;self.repair_frame=None
        if event is not None:
            _,contract=validate(event)
            event=copy.deepcopy(event)
            frame=event['data']['frame']
            frame.update(repair_contract='base_owner_only',risk_protected=contract['protected'],
                         risk_weight_caps=contract['weight_caps'])
            if event['received_ms']/1000<=now and 0<=now-frame['source_time']<21600:
                self.repair_frame=frame
        if self.base_task and now>=self.base_task['expires']:
            self.events.append(dict(time=now,kind='pending_rebalance_expired',expires=self.base_task['expires']))
            self.base_task=None
        active=self.plan['paper_start_unix']<=now<self.plan['capture_end_unix']-self.plan['end_liquidation_seconds']
        if self.repair_frame and active and now>=self.ready_at and not self.account.unknown and not self.halted:
            self.settlement(now*1000,self.account)
            marks,quotes=self.prices(now,streams)
            if marks is not None:
                equity=self.account.reconcile(marks);sigma=np.asarray(frame['sigma'])
                forced=bool(risk_breaches(self.account.physical.qty,equity,marks,sigma,self.p['risk']))
                forced|=equity/max(self.peak,equity)-1<=-self.p['risk']['drawdown_halt']
                if not forced:
                    self.peak=max(self.peak,equity)
                    held=self.account.owners['base'].qty.copy()
                    target,_=self.protected_target(held,marks,equity)
                    # Reproject pending intent on today's caps, never increase it
                    # again just because a later model publication loosens a cap.
                    if self.base_task:
                        self.base_task['target'],_=self.protected_target(self.base_task['target'],marks,equity)
                    target=self.defer_dust(target,marks,quotes,frame['source_time'])
                    if np.any(abs(target-held)>1e-10):
                        self.events.append(dict(time=now,kind='risk_cap_request',source=frame['source_time'],
                            prior_qty=held.tolist(),target_qty=target.tolist(),execution_revision=REVISION,
                            pending_target_preserved=bool(self.base_task),
                            pending_expires=self.base_task['expires'] if self.base_task else None))
                        # Keep the task. Confirmed partial fills change holdings;
                        # next tick trims only the remainder before resumption.
                        # Already capped and dust-filtered: a second cap projection
                        # would reintroduce the deferred subminimum reductions.
                        Coordinator.submit_baseline(self,now,target,streams,marks,sigma,True)
                        return
        # Bypass only RepairPortfolio's destructive model-cap branch. Validation
        # above and submit_baseline keep its risk-cap contract; the coordinator's
        # unknown-outcome guard, global/terminal stops and pair lifecycle remain.
        return Coordinator.step(self,now,streams,event,forecasts)
