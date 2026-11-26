"""PAS quote-mid pullback features with original signed alpha/risk logic.

Trade inactivity is not quote inactivity. Missing historical quotes still veto
additions; they are never filled using a future observation or reference price.
"""
import copy
import numpy as np
from candidate_handoff.portfolio import CandidatePortfolio,target_from_frame
from candidate_handoff.producer import validate
from candidate_handoff.entry_policy import BASE
from candidate_handoff.risk_priority import RiskPriorityPortfolio
from strategy_final_risk.candidate import publication
from experiments.basis_carry_20261017.probe import asof
from replay_lab.common_comparison.engine import project,rounded,risk_breaches
from .quote_guard import QuotePortfolio,QuoteAccount,freeze_sides
from execution_revision_20261024.portfolio import reduction_target


def quote_history(streams,symbols,source,cfg):
    points=[source-86400,source-10800,source]
    prices=np.full((3,len(symbols)),np.nan);evidence=[]
    for i,point in enumerate(points):
        for j,symbol in enumerate(symbols):
            s=streams.get(symbol);q=asof(s,point*1000,cfg) if s is not None else None
            item=dict(point=point,symbol=symbol,valid=q is not None)
            if q is not None:
                prices[i,j]=q['mid'];item.update(mid=q['mid'],received=q['received'],server_time=q['book']['time'])
            evidence.append(item)
    return prices,evidence


def filter_target(target,held,prices):
    target=np.asarray(target,float).copy();held=np.asarray(held,float)
    valid=np.isfinite(prices)&(prices>0)
    ok=np.zeros(len(target),bool)
    for i in [0,1]:
        good=valid[i]&valid[2]
        delta=np.log(np.divide(prices[2],prices[i],out=np.ones(len(target)),where=good))
        ok |= good & (np.sign(target)*delta<=0)
    for j in range(len(target)):
        if ok[j]:continue
        if target[j]*held[j]<=0:target[j]=0.
        elif abs(target[j])>abs(held[j]):target[j]=held[j]
    return target,ok


def quote_target(frame,marks,qty,equity,protocol,streams,cfg):
    basic=copy.deepcopy(frame);basic['execution_policy']=BASE
    action,trace=target_from_frame(basic,marks,qty,equity,protocol)
    if action is None:return None,trace
    original=np.asarray(action['target_qty'],float)
    prices,evidence=quote_history(streams,protocol['symbols'],frame['source_time'],cfg)
    target,ok=filter_target(original,qty,prices)
    weights=project(target*marks/equity,equity,np.asarray(frame['sigma']),protocol['risk'])
    q=np.array([rounded(v,n) for v,n in zip(weights*equity/marks,protocol['execution']['decimals'])])
    if np.any(abs(q)>abs(original)+1e-10):raise AssertionError('Quote filter restored a reduced quantity')
    if risk_breaches(q,equity,marks,np.asarray(frame['sigma']),protocol['risk']):raise AssertionError('Final risk violation')
    action['target_qty']=q.tolist()
    trace.update(entry_history_kind='received_pas_quote_mid',entry_filter_pass=ok.tolist(),
                 pre_entry_filter_qty=original.tolist(),target_qty=q.tolist(),quote_history=evidence)
    return action,trace


class QuoteHistoryPortfolio(QuotePortfolio):
    """Modes isolate the history feature from the separately tested guard."""
    def submit_baseline(self,*args,**kwargs):
        if self.quote_mode=='history_only':return RiskPriorityPortfolio.submit_baseline(self,*args,**kwargs)
        return QuotePortfolio.submit_baseline(self,*args,**kwargs)

    def step_candidate(self,now,streams,packet,profile):
        if packet is None:return RiskPriorityPortfolio.step(self,now,streams,None,[])
        body=validate(packet,profile,now,expected_policy=getattr(self,'execution_policy',BASE))
        f=body['frame'];old=body['original_signal']
        event=publication(old['publication'],f['anchor_weights'],np.array(f['protected'],bool))
        if self.quote_mode!='history_only' and self.base_task and now>=self.base_task.get('entry_expires',self.base_task['expires']):
            task=self.base_task
            if 'intent_source' not in task or now>=task['intent_source']+21600:self.base_task=None
            else:
                held=self.account.owners['base'].qty;task['target']=reduction_target(held,task['target'])
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
                    action,trace=quote_target(f,marks,self.account.owners['base'].qty,equity,self.p,streams,self.cfg)
                    if action is not None:
                        life=900 if self.quote_mode=='history_only' else 3600
                        self.base_task=dict(target=np.array(action['target_qty']),expires=boundary+life,
                            entry_expires=boundary+life,intent_source=boundary,reductions_only=False,
                            decision_prices=marks.copy(),frozen_side_prices=freeze_sides(self.p['symbols'],quotes))
                        self.last_decision=boundary
                        self.events.append(dict(time=now,source=f['source_time'],kind='quote_history_decision',
                            trace=trace,target_qty=action['target_qty'],signal_sha256=packet['sha256'],quote_mode=self.quote_mode))
        return RiskPriorityPortfolio.step(self,now,streams,event,old['forecasts'])


class Activated(CandidatePortfolio):
    def step_candidate(self,now,streams,packet,profile):
        if now<self.activation:return CandidatePortfolio.step_candidate(self,now,streams,packet,profile)
        if self.quote_mode!='history_only':
            self.account.__class__=QuoteAccount;self.account.frozen_side_prices=None
        self.__class__=QuoteHistoryPortfolio;self.base_task=None
        return self.step_candidate(now,streams,packet,profile)
