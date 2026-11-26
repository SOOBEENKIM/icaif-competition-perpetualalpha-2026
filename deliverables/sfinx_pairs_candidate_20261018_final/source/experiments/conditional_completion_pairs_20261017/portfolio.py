"""Adaptive mean experiment; derivative of own frozen execution engine."""
from experiments.paid_rehedge_pairs_20261017.policy import held_state,renewal_plan,anchor_stop,reconcile_adjustment
from experiments.historical_par_extension_20261017.settlements import charge
from experiments.dense_history_pairs_20261017.policy import usable_signals
from experiments.dense_history_pairs_20261017.policy import proposals
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from .completion import completion_plan, after_completion
import copy
import numpy as np
from replay_lab.common_comparison.models import Market
from replay_lab.common_comparison.engine import Ledger,execute_intent as original_execute_intent,risk_breaches,risk_values,rounded
from replay_lab.pullback.broker import execution_price
from replay_lab.ema_rebuild.engine import Controller
from experiments.cost_horizon_20261016.research import SecondController
from experiments.basis_carry_20261017.score import score


from experiments.quote_anchored_pairs_20261017.pricing import execute_intent, reservation_limits


def replay(d,de,pred,cfg,protocol,arm,scenario,start,end,*,overlay=False):
    p=copy.deepcopy(protocol);sc=cfg['scenarios'][scenario];p['execution']['fee_bps']=sc['fee_bps']
    t=d['times'];step=int(t[1]-t[0]);assert np.all(np.diff(t)==step) and step in (300,3600)
    m=Market(t,d['par_close'],d['par_volume'],d['par_trades'],d['funding'],d['funding_observed'],tuple(p['symbols']))
    first,last=m.exact(start),m.exact(end);account=SharedAccount(p['initial'],m.symbols)
    ledger=account.physical;base_ledger=account.owners['base'];pair_ledger=account.owners['pair']
    base=arm.startswith(('first','second'))
    has_pair=arm not in ('first_strategy','second_strategy')
    h=int(arm.rsplit('_',1)[1]) if '_dense_' in arm else (168 if has_pair else None)
    complete_enabled=True
    rehedge_enabled="_renewhedge_" in arm or "_hourlyhedge_" in arm
    renewal_enabled="_renew_" in arm or rehedge_enabled;renewals=0
    adjustments=adjusted=adjustment_unfilled=adjustment_failed=0
    control=Controller() if arm.startswith('first') else SecondController(t,d['retention'])
    raw=d['first'] if arm.startswith('first') else d['second']
    pending=None;task=None;pair=None;snapshots=[];orders=[];events=[];decisions=[];locks={};used={};attempts={}
    peak=ledger.initial;mdd=0.;halted=False;accounting=0.;audit_eq=ledger.initial
    price_move=impact=0.;aborts=repairs=completed=0;breaches=[];unbalanced=0
    for i in range(first,last+1):
        now=int(t[i]);prices=m.close[i];sigma=d['sigma'][i];prior=ledger.qty.copy()
        if i>first:
            move=float(prior@(prices-m.close[i-1]));price_move+=move;audit_eq+=move
            if 'funding_cutover' in d and now>int(d['funding_cutover']):
                stamp=int(d['funding_event_ms'][i])
                if stamp>=0:
                    audit_eq+=charge(account,now,d['funding_prices'][i],m.funding[i],m.observed[i],stamp,sc['funding_receipt_multiplier'])
            elif now%3600==0:
                rate=m.funding[i].copy();cash=-prior*d['reference_close'][i]*rate
                rate=np.where(cash>0,rate*sc['funding_receipt_multiplier'],rate)
                account.charge_funding(now,d['reference_close'][i],rate,m.observed[i])
                audit_eq-=float(prior@(d['reference_close'][i]*rate))
        nf=len(ledger.fills)
        if pending is not None:
            pending['sigma']=sigma
            orders.extend(account.execute_candle(pending,m,i,p,sc))
            if pending.get('pair_entry'):
                q=pair_ledger.qty[pair['ids']];want=np.array(pair['target']);ratios=q/want
                if np.all(q==0):
                    key=(pair['pair_index'],pair['forecast_source'])
                    events.append(dict(time=now,event='entry_unfilled',pair=pair['pair'],forecast_source=pair['forecast_source'],attempt=attempts[key]))
                    if attempts[key]<cfg['maximum_zero_fill_attempts']:used.pop(pair['pair_index'],None)
                    pair=None
                else:
                    pair.update(first_fill=now,exit_time=pair['forecast_source']+pair['horizon']*3600)
                    if min(ratios)<=0:
                        if complete_enabled:
                            pair.update(completing=True,completion_deadline=now+step)
                            events.append(dict(time=now,event='one_leg_completion_pending',pair=pair['pair']))
                        else:
                            pair.update(closing=True,aborted=True);aborts+=1;events.append(dict(time=now,event='one_leg_abort',pair=pair['pair']))
                    elif abs(ratios[0]-ratios[1])/max(ratios)>cfg['pair_balance_tolerance']:
                        repair=np.array([rounded(x*min(ratios),p['execution']['decimals'][j]) for j,x in zip(pair['ids'],want)])
                        pair.update(repairing=True,repair_deadline=now+cfg['repair_deadline_seconds'],repair_qty=repair.tolist())
                        if np.any(repair==0):pair.update(repairing=False,closing=True,aborted=True);aborts+=1
                        events.append(dict(time=now,event='repair_requested',pair=pair['pair'],target=repair.tolist()))
                    else:pair['balanced']=True;events.append(dict(time=now,event='balanced_open',pair=pair['pair']))
            elif pending.get('completion_order'):
                outcome=after_completion(pair,pair_ledger.qty,p,cfg,now)
                if outcome=='abort':aborts+=1
                events.append(dict(time=now,event='completion_'+outcome,pair=pair['pair']))
            elif pending.get('rehedge_order'):
                arrival_equity=ledger.equity(prices)
                blocked=halted or arrival_equity/max(peak,arrival_equity)-1<=-p['risk']['drawdown_halt']
                outcome=reconcile_adjustment(pair,pair_ledger,i,d,de,cfg,p,sc,ledger.qty,arrival_equity,
                    all_outcomes_observed=True,risk_blocked=blocked)
                events.append(outcome)
                if outcome['outcome']=='renewed':
                    adjusted+=1;renewals+=1;used[pair['pair_index']]=outcome['new_source']
                elif outcome['outcome']=='unfilled':adjustment_unfilled+=1
                else:adjustment_failed+=1
            pending=None
        for f in ledger.fills[nf:]:
            c=f['signed_qty']*(f['price']-prices[f['asset']]);impact+=c;audit_eq-=c+f['fee']
        equity=account.reconcile(prices);accounting=max(accounting,abs(equity-audit_eq));assert accounting<1e-6
        if pair:
            q=pair_ledger.qty[pair['ids']]
            if pair.get('closing') and np.all(q==0):
                if pair.get('balanced'):completed+=1
                events.append(dict(time=now,event='closed',pair=pair['pair'],balanced=pair.get('balanced',False),aborted=pair.get('aborted',False)))
                pair=None
            elif pair:
                ratios=q/np.array(pair['target'])
                is_balanced=min(ratios)>0 and abs(ratios[0]-ratios[1])/max(ratios)<=cfg['pair_balance_tolerance']
                unbalanced+=int(np.any(q) and not is_balanced)
                if pair.get('repairing') and not pair.get('closing'):
                    if is_balanced:
                        pair.update(repairing=False,balanced=True);repairs+=1
                        events.append(dict(time=now,event='balanced_after_repair',pair=pair['pair']))
                    elif now>=pair['repair_deadline']:
                        pair.update(repairing=False,closing=True,aborted=True);aborts+=1
                        events.append(dict(time=now,event='repair_timeout',pair=pair['pair']))
                if not pair.get('closing') and pair.get('balanced'):
                    if pair.get('dense') and 'risk_entry_prices' not in pair:
                        pair['risk_entry_prices']=pair_ledger.entry[pair['ids']].copy().tolist()
                        pair['risk_quantities']=pair_ledger.qty[pair['ids']].copy().tolist()
                    if rehedge_enabled:anchor_stop(pair,pair_ledger,prices)
                    state=held_state(pair,i,d,de,cfg,ledger=pair_ledger)
                    if renewal_enabled and state['reason'] is None:
                        blocked=halted or equity/max(peak,equity)-1<=-p['risk']['drawdown_halt']
                        plan=renewal_plan(pair,pair_ledger.qty,i,d,de,cfg,p,sc,ledger.qty,equity,risk_blocked=blocked,ledger=pair_ledger)
                        if plan is not None:
                            kind='hedge_adjustment_planned' if plan.get('requires_adjustment') else ('renewed' if plan['accepted'] else 'renewal_rejected')
                            events.append(dict(time=now,event=kind,pair=pair['pair'],**plan))
                            if plan.get('requires_adjustment'):
                                pair.update(rehedging=True,rehedge_plan=plan,last_rehedge_attempt_source=plan['new_source'])
                            elif plan['accepted']:
                                pair['renewal_maturity']=plan['new_maturity'];pair['latest_renewal_source']=plan['new_source']
                                used[pair['pair_index']]=plan['new_source'];renewals+=1
                                state=held_state(pair,i,d,de,cfg,ledger=pair_ledger)
                    if state['reason']:
                        pair['closing']=True
                        events.append(dict(time=now,event='exit_requested',pair=pair['pair'],**state))
                        if state['reason'] in ('spread_stop','time_stop'):locks[pair['pair_index']]=copy.deepcopy(pair)
        peak=max(peak,equity);mdd=min(mdd,equity/peak-1);halted|=equity/peak-1<=-p['risk']['drawdown_halt']
        breach=risk_breaches(ledger.qty,equity,prices,sigma,p['risk'])
        if breach:breaches.append(dict(time=now,limits=breach))
        if i<last:
            action=False;entry=False;completion=None;adjustment=None;forced=bool(breach) or halted
            base_target=base_ledger.qty.copy();pair_target=pair_ledger.qty.copy()
            if forced:
                task=None;base_target[:]=0.;pair_target[:]=0.;action=True
                if pair:pair['closing']=True
            else:
                if base:
                    window=3600//step
                    past_volume=m.volume[max(0,i-window+1):i+1].sum(0);past_trades=m.trades[max(0,i-window+1):i+1].sum(0)
                    proposed=control.propose(now,prices,base_ledger.qty,equity,raw[i],sigma,past_volume,past_trades,p,buffer=.2)
                    if proposed is not None:task=dict(target=np.array(proposed['target_qty']),expires=now+cfg['intent_deadline_seconds'])
                    if task and now<task['expires']:base_target=task['target'].copy();action=True
                    elif task:task=None
                if pair:
                    if pair.get('closing'):pair_target[pair['ids']]=0.;action=True
                    elif pair.get('rehedging'):
                        proposed_pair=pair_target.copy();proposed_pair[pair['ids']]=pair['rehedge_plan']['target']
                        if risk_breaches(base_target+proposed_pair,equity,prices,sigma,p['risk']):
                            pair.update(rehedging=False)
                            events.append(dict(time=now,event='adjustment_packet_rejected',pair=pair['pair'],reason='combined_base_target_risk'))
                            pair.pop('rehedge_plan')
                        else:
                            adjustment=pair['rehedge_plan'];pair_target=proposed_pair;action=True;adjustments+=1
                    elif pair.get('completing'):
                        completion=completion_plan(pair,pair_ledger,i,d,de,cfg,p,sc,risk_ledger=ledger)
                        if completion is None:
                            pair.update(completing=False,closing=True,aborted=True);aborts+=1
                            pair_target[pair['ids']]=0.;action=True
                            events.append(dict(time=now,event='completion_rejected_unwind',pair=pair['pair']))
                        else:
                            pair_target[pair['ids']]=completion['target'];action=True
                            events.append(dict(time=now,event='completion_intent',pair=pair['pair'],**completion))
                    elif pair.get('repairing'):pair_target[pair['ids']]=pair['repair_qty'];action=True
                elif has_pair and now%cfg['signal_cadence_seconds']==0:
                    directions=usable_signals(i,d,de,cfg,'fixed_delayed_immediate_complete_168',locks,used)
                    choices=proposals(i,d,de,pred,h,cfg,p,sc,pair_ledger.qty,equity,np.zeros(10),directions,arm,risk_qty=ledger.qty)
                    for proposal in choices:
                        check=pair_target.copy();check[proposal['ids']]=proposal['target']
                        if risk_breaches(base_target+check,equity,prices,sigma,p['risk']):continue
                        pair=dict(proposal,closing=False,repairing=False,balanced=False)
                        if renewal_enabled:pair['renewable']=True
                        if rehedge_enabled:pair['paid_rehedge_enabled']=True
                        used[proposal['pair_index']]=proposal['forecast_source']
                        key=(proposal['pair_index'],proposal['forecast_source']);attempts[key]=attempts.get(key,0)+1
                        pair_target=check;action=True;entry=True;events.append(dict(time=now,event='pair_intent',**proposal));break
            if action:
                pending=dict(decision_time=now,execution_time=now+step,source_close=now,source_available=now,
                    source_volume=m.volume[i].copy(),source_trades=m.trades[i].copy(),source_prices=prices.tolist(),
                    owner_targets={'base':base_target.tolist(),'pair':pair_target.tolist()},target_qty=(base_target+pair_target).tolist(),risk_forced=forced)
                if entry:
                    pending['pair_entry']=True
                    pending['owner_limits']={'pair':pair['entry_limits']}
                    pending['expected_net_cash']=pair['expected_net_cash']
                if completion is not None:
                    pending.update(completion_order=True,owner_limits={'pair':completion['entry_limits']})
                if adjustment is not None:
                    pending.update(rehedge_order=True,owner_limits={'pair':adjustment['entry_limits']})
                pending=account.seal(pending,p,sc)
                decisions.append({k:v for k,v in pending.items() if k!='sealed_owner_limits'})
        snapshots.append(dict(time=now,equity=equity,cash=ledger.cash,qty=ledger.qty.tolist(),fees=ledger.fees,
            funding=ledger.funding,base_qty=base_ledger.qty.tolist(),pair_qty=pair_ledger.qty.tolist(),**risk_values(ledger.qty,equity,prices,sigma)))
    days=[s for s in snapshots if s['time']%86400==43200];values=np.array([s['equity'] for s in days]);returns=values[1:]/values[:-1]-1
    assert days[0]['time']==start and days[-1]['time']==end
    roi=equity/ledger.initial-1;td=len({f['time']//86400 for f in ledger.fills});financial=score(roi,returns,trading_days=td,start_sample=True)
    gross=float(abs(ledger.qty)@prices);reserve=gross*(sc['fee_bps']+sc['impact_bps'])/10000
    metrics=dict(base_owned_pnl=base_ledger.equity(prices)-base_ledger.initial,pair_owned_pnl=pair_ledger.equity(prices),internal_transfers=len(account.transfers),maximum_ownership_error=account.maximum_error,net_pnl=equity-ledger.initial,roi=roi,**financial,fees=ledger.fees,funding=ledger.funding,price_move_pnl=price_move,
        execution_cost=impact,max_drawdown=-mdd,mean_gross=float(np.mean([s['gross'] for s in snapshots])),maximum_gross=max(s['gross'] for s in snapshots),
        fills=len(ledger.fills),orders=len(orders),completed_pairs=completed,aborted_pairs=aborts,hedge_repairs=repairs,
        unbalanced_minutes=unbalanced*step/60,residual_gross=gross,liquidation_cost_estimate=reserve,
        pnl_after_liquidation_reserve=equity-ledger.initial-reserve,maximum_accounting_error=max(accounting,ledger.max_error),risk_breach_bars=len(breaches))
    if renewal_enabled:metrics['renewals']=renewals
    if rehedge_enabled:
        metrics.update(hedge_adjustment_intents=adjustments,hedge_adjustments_completed=adjusted,
            hedge_adjustments_unfilled=adjustment_unfilled,hedge_adjustments_failed=adjustment_failed)
    assert abs(np.prod(1+returns)-1-roi)<1e-10
    assert abs(metrics['net_pnl']-price_move+impact+ledger.fees-ledger.funding)<1e-6
    assert all(f['source_available']<=f['decision_time']<f['time'] for f in ledger.fills)
    return dict(arm=arm,overlay=overlay,scenario=scenario,start=start,end=end,metrics=metrics,
        internal_transfers=account.transfers,owner_fills={k:v.fills for k,v in account.owners.items()},snapshots=snapshots,orders=orders,decisions=decisions,fills=ledger.fills,events=events,funding_events=ledger.funding_events,
        daily=[dict(time=x['time'],return_=float(r)) for x,r in zip(days[1:],returns)],risk_breaches=breaches,orders_sent=0)
