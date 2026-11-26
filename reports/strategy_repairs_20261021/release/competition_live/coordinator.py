"""Additive retained-gate adapter; original native and direct paths preserved.

PAS deployment copy of the retained gate coordinator. Polling may arrive in
the first 30 seconds of a 15-minute entry bucket; timestamps stay actual.
Fresh marks value inactive markets; only executable books can generate fills.
"""
import copy
import math
import numpy as np
from experiments.basis_carry_20261017.probe import asof
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry
from replay_lab.ema_rebuild.engine import Controller
from experiments.cost_horizon_20261016.research import SecondController
from replay_lab.common_comparison.engine import risk_breaches, risk_values
from experiments.published_pair_lifecycle_20261017.account import BookAccount
from experiments.published_pair_lifecycle_20261017.lifecycle import PairLifecycle


class Portfolio:
    def __init__(self, name, scenario, plan, settlement):
        self.name, self.plan = name, plan
        self.p, self.cfg = copy.deepcopy(plan['trading_protocol']), copy.deepcopy(plan['book_protocol'])
        self.sc = {**self.cfg['scenarios'][scenario], 'impact_bps':0.}
        self.cfg['exit_lead_seconds'] = plan.get('step_seconds',30)+math.ceil((self.sc['latency_ms']+self.cfg['future_exit_observation_grace_ms'])/1000)
        self.cfg['latest_maturity_unix'] = plan['capture_end_unix']-plan['end_liquidation_seconds']
        self.account = BookAccount(self.p['initial'], self.p['symbols'])
        self.settlement = settlement
        self.pair = None; self.sessions = []; self.events = []; self.snapshots = []
        self.last_decision = -1; self.base_task = None; self.ready_at = 0; self.counter = 0
        self.used = {}; self.attempts = {}; self.pair_bars = set()
        self.peak = self.p['initial']; self.halted = False; self.fatal = []
        self.model = {'first_plus_direct':'direct','first_plus_hourly_bias':'bias',
                      'first_plus_decision_gate':'decision_gate'}.get(name)

    def prices(self, now, streams):
        quotes = {s:asof(streams[s],now*1000,self.cfg) if s in streams else None for s in self.p['symbols']}
        # Valuation needs fresh mark contexts, not executable depth in every
        # unrelated market. Entry/exit previews still require their own books.
        marks=[];self.unpriced_flat_assets=[]
        for j,symbol in enumerate(self.p['symbols']):
            rows=streams.get(symbol,{}).get('ctx',[])
            eligible=[row for row in rows if 0<=now*1000-row[0]<=self.cfg['context_max_age_ms']]
            try:mark=float(eligible[-1][2]['markPx']) if eligible else float('nan')
            except (KeyError,TypeError,ValueError):mark=float('nan')
            if not np.isfinite(mark) or mark<=0:
                # Offset owner positions still need a real mark even at net zero.
                if any(ledger.qty[j]!=0 for ledger in [self.account.physical,*self.account.owners.values()]):
                    return None,quotes
                # Explicit zero-notional sentinel, NEVER an executable price.
                # New exposure is disabled until all marks are available.
                mark=0.;self.unpriced_flat_assets.append(symbol)
            marks.append(mark)
        return np.asarray(marks),quotes

    def submit_baseline(self, now, target, streams, marks, sigma, forced):
        self.counter += 1
        raw=dict(packet_id=f'base:{self.counter}',decision_time=now,source_close=now,source_available=now,
            source_prices=marks.tolist(),risk_forced=forced,owner_targets={'base':target.tolist(),
                'pair':self.account.owners['pair'].qty.tolist()},sigma=sigma)
        packet=self.account.seal(raw,self.p,self.sc)
        result=self.account.execute_books(packet,streams,marks,sigma,self.p,self.sc,self.cfg,on_time=self.settlement)
        self.events.append(dict(time=now,kind='baseline_packet',result=result))
        self.ready_at=math.ceil(max([now*1000,self.account.last_packet_arrival_ms])/1000)+1

    def step(self, now, streams, publication, forecasts):
        start,end=self.plan['paper_start_unix'],self.plan['capture_end_unix']
        if now<start or now>end or now<self.ready_at:
            return
        if self.account.unknown:
            if not self.fatal:self.fatal.append('unobserved_order_outcome')
            return
        self.settlement(now*1000,self.account)
        marks,quotes=self.prices(now,streams)
        if marks is None:
            self.fatal.append(f'missing_account_valuation:{now}')
            return
        equity=self.account.reconcile(marks)
        self.peak=max(self.peak,equity)
        self.halted |= equity/self.peak-1<=-self.p['risk']['drawdown_halt']
        frame=publication['data']['frame'] if publication else None
        fresh=(frame is not None and publication['received_ms']/1000<=now
               and 0<=now-frame['source_time']<6*3600)
        sigma=np.asarray(frame['sigma']) if frame is not None else np.eye(len(marks))*1e-6
        terminal=now>=end-self.plan['end_liquidation_seconds']
        forced=terminal or self.halted or not fresh or bool(risk_breaches(self.account.physical.qty,equity,marks,sigma,self.p['risk']))
        if self.pair and self.pair.state in ('closed','unfilled','rejected'):
            if self.pair.state=='unfilled':
                key=(tuple(self.pair.intent['pair']),self.pair.intent['forecast_source'])
                if self.attempts.get(key,0)<self.cfg['maximum_zero_fill_attempts']:
                    self.used.pop(key[0],None)
            self.sessions.append(self.pair);self.pair=None
        base=self.account.owners['base'].qty.copy()
        if forced:
            self.base_task=None;base[:]=0.
        elif fresh and not self.unpriced_flat_assets:
            boundary=now//21600*21600
            if boundary>self.last_decision and now-boundary<=900 and frame['source_time']==now//3600*3600:
                if self.name=='second_strategy':
                    control=SecondController([boundary],[frame['retention']]);control.controller.last_decision=self.last_decision
                    raw=publication['data']['candidate']
                else:
                    control=Controller(last_decision=self.last_decision);raw=publication['data']['incumbent']
                action=control.propose(boundary,marks,base,equity,np.array(raw),sigma,
                    np.array(frame['volume']),np.array(frame['trades']),self.p,buffer=.2)
                if action is not None:
                    self.base_task=dict(target=np.array(action['target_qty']),expires=boundary+900)
                    self.last_decision=boundary
            if self.base_task and now<self.base_task['expires']:
                base=self.base_task['target'].copy()
            else:self.base_task=None
        else:
            self.base_task=None
        # Snapshot is before this tick's future arrival fills, never after them
        # while retaining a pre-arrival timestamp.
        self.snapshots.append(dict(time=now,equity=equity,cash=self.account.physical.cash,
            fees=self.account.physical.fees,funding=self.account.physical.funding,
            qty=self.account.physical.qty.tolist(),**risk_values(self.account.physical.qty,equity,marks,sigma)))
        if now==end:return
        submitted=False
        if self.pair:
            self.pair.baseline_target=base.copy()
            self.pair.risk_peak=self.peak
            current=next((f for f in forecasts if f['model']==self.model and f['pair']==self.pair.intent['pair']
                          and f['available_time']<=now),None)
            before=len(self.pair.events)
            self.pair.advance(now,streams,marks,sigma,forecast=current,
                force_reason=('evaluation_end' if terminal else 'account_risk_or_stale_signal') if forced else None)
            submitted=any(e.get('result') is not None for e in self.pair.events[before:])
            self.ready_at=max(self.ready_at,self.pair.ready_at)
        elif self.model and not forced and not self.unpriced_flat_assets:
            cadence=self.cfg['signal_cadence_seconds'];bar=now//cadence
            if now%cadence<30 and bar not in self.pair_bars:
                self.pair_bars.add(bar);choices=[]
                for f in forecasts:
                    if f['model']!=self.model or not f['forecast_source']<=f['available_time']<=now:
                        continue
                    if f['forecast_source']+f['horizon']*3600>end-self.plan['end_liquidation_seconds']:
                        continue
                    if self.used.get(tuple(f['pair']),-1)>=f['forecast_source']:
                        continue
                    for direction in (-1,1):
                        # Only NEW entry proposals are gated. The existing
                        # holding branch above still receives direct forecasts.
                        if self.model=='decision_gate' and (not f.get('economic_available',False)
                                or f.get('economic_action')!=direction):
                            self.events.append(dict(time=now,kind='entry_check',pair=f['pair'],
                                source=f['forecast_source'],direction=direction,reason='decision_gate_rejected'))
                            continue
                        intent=dict(f,direction=direction,account_marks=marks.tolist())
                        preview=plan_entry(intent,quotes,now,self.cfg,self.p,self.sc,self.account.physical.qty,
                            equity,sigma,owned_qty=self.account.owners['pair'].qty)
                        self.events.append(dict(time=now,kind='entry_check',pair=f['pair'],source=f['forecast_source'],direction=direction,reason=preview['reason']))
                        if preview['accepted']:choices.append((preview,intent))
                if choices:
                    preview,intent=max(choices,key=lambda x:(x[0]['expected_net_cash'],x[1]['pair']))
                    self.pair=PairLifecycle(self.account,intent,self.cfg,self.p,self.sc,on_time=self.settlement)
                    self.pair.baseline_target=base.copy()
                    self.pair.risk_peak=self.peak
                    self.pair.enter(now,streams,marks,sigma)
                    key=(tuple(intent['pair']),intent['forecast_source'])
                    self.attempts[key]=self.attempts.get(key,0)+1;self.used[key[0]]=key[1]
                    submitted=self.pair.state!='rejected'
                    self.ready_at=max(self.ready_at,self.pair.ready_at)
        if not submitted and not self.account.unknown and np.any(abs(base-self.account.owners['base'].qty)>1e-10):
            self.submit_baseline(now,base,streams,marks,sigma,forced)
