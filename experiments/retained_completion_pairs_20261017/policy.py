"""One observed, paid completion of a risk-reducing retained-size packet."""
import copy
import numpy as np
from experiments.retained_sizing_pairs_20261017.policy import (
    anchor_stop, held_state, renewal_plan, value_at, exit_value,
    reconcile_adjustment as prior_reconcile,
)
from replay_lab.common_comparison.engine import risk_breaches


def reduces_toward(target, old):
    target=np.asarray(target);old=np.asarray(old)
    return bool(np.all(target*old>0) and np.all(abs(target)<=abs(old)+1e-10))


def completion_plan(pair,previous,ledger,i,d,cfg,p,sc,account_qty,equity,*,risk_blocked=False):
    def reject(reason):return dict(accepted=False,reason=reason)
    now=int(d['times'][i]);step=int(d['times'][1]-d['times'][0]);ids=list(pair['ids'])
    if previous.get('retained_completion_attempt',0)>=1:return reject('completion_attempt_exhausted')
    target=np.array(previous['target']);original=np.array(previous['old_qty']);owned=ledger.qty[ids].copy()
    if not reduces_toward(target,original) or not reduces_toward(target,owned):return reject('remaining_packet_not_reduction_only')
    state=held_state(pair,i,d,None,cfg,ledger=ledger)
    if state['reason'] is not None:return reject('original_stop_or_exit_active')
    deadline=min(previous['old_maturity'],previous['new_maturity'],previous['hard_holding_deadline'])
    if now+step>=deadline:return reject('completion_would_miss_original_deadline')
    marks=d['par_close'][i];sigma=d['sigma'][i]
    if risk_blocked or risk_breaches(account_qty,equity,marks,sigma,p['risk']):return reject('completion_global_risk_or_halt')
    delta=target-owned;active=np.flatnonzero(abs(delta)>1e-10)
    if not len(active):return reject('nothing_to_complete')
    terminal=np.array(previous['terminal_prices']);maturity=previous['new_maturity'];gamma=previous['risk_aversion']
    value=value_at(target,owned,account_qty,ids,i,d,maturity,terminal,p,sc,gamma,equity)
    immediate=exit_value(owned,marks[ids],terminal,ids,i,d,maturity,p,sc)['immediate_close_cash']
    gross=float(abs(target)@marks[ids]);buffer=gross*cfg['renewal_policy']['incremental_buffer_bps']/10000
    if gross>min(previous['original_balanced_gross'],cfg['pair_max_gross'])+1e-8:return reject('retained_gross_exceeds_original_budget')
    cash=value['expected_hold_cash']-value['adjustment_cost_preview']-immediate
    utility=value['utility']-immediate
    if not np.isfinite([cash,utility]).all() or min(cash,utility)<=buffer:return reject('remaining_completion_value_insufficient')
    preview=np.array(value['preview'])
    if any(abs(delta[n])*preview[n]<p['execution']['minimum_order_value'] for n in active):return reject('remaining_delta_below_minimum_order')
    slack=min(cash,utility)-buffer;limits=preview.copy();fee=sc['fee_bps']/10000
    for n in active:
        side=np.sign(delta[n]);distance=slack/len(active)/(abs(delta[n])*(1+side*fee))
        limits[n]+=side*min(distance,limits[n]*.99)
    worst=float(delta@(limits-marks[ids])+abs(delta)@limits*fee)
    worst_cash=value['expected_hold_cash']-worst-immediate
    worst_utility=worst_cash-value['risk_charge']
    assert min(worst_cash,worst_utility)>=buffer-1e-8
    for mask in ([1,0],[0,1],[1,1]):
        q=np.asarray(account_qty).copy();q[ids]+=delta*np.array(mask)
        if risk_breaches(q,equity-worst,marks,sigma,p['risk']):return reject('completion_independent_leg_risk')
    return dict(previous,accepted=True,requires_adjustment=True,reason='costed_partial_reduction_completion',
        old_qty=owned.tolist(),target=target.tolist(),delta=delta.tolist(),target_gross=gross,current_gross=float(abs(owned)@marks[ids]),
        adjustment_cost_preview=value['adjustment_cost_preview'],immediate_close_cash=immediate,
        expected_hold_cash=value['expected_hold_cash'],expected_funding_cash=value['expected_funding_cash'],
        expected_price_cash=value['expected_price_cash'],estimated_future_exit_cost=value['estimated_future_exit_cost'],
        risk_charge=value['risk_charge'],original_renewal_utility=immediate,retained_utility=value['utility'],
        utility_advantage=utility,utility_buffer_cash=buffer,incremental_advantage=cash,buffer_cash=buffer,
        entry_limits={str(ids[n]):float(limits[n]) for n in active},minimum_advantage_at_limits=worst_cash,
        minimum_utility_advantage_at_limits=worst_utility,orders_required=len(active),decision_time=now,
        retained_completion_attempt=1,retained_completion_deadline=now+step,
        initial_adjustment_decision_time=previous['decision_time'],
        initial_sealed_limits=previous['entry_limits'],comparison_basis='paid close of actual partial inventory',
        first_packet_costs_already_in_ledger=True)


def reconcile_adjustment(pair,ledger,i,d,de,cfg,p,sc,account_qty,equity,*,all_outcomes_observed,risk_blocked=False):
    previous=pair.get('rehedge_plan',{})
    enabled=pair.get('retained_completion_enabled') and previous.get('utility_enabled')
    if not enabled or not all_outcomes_observed:
        return prior_reconcile(pair,ledger,i,d,de,cfg,p,sc,account_qty,equity,
            all_outcomes_observed=all_outcomes_observed,risk_blocked=risk_blocked)
    # Reconciliation mutates only pair state, not ledger cash or actual fills.
    # Probe on a copy so a valid completion can retain the live partial state.
    probe=copy.deepcopy(pair)
    outcome=prior_reconcile(probe,ledger,i,d,de,cfg,p,sc,account_qty,equity,
        all_outcomes_observed=True,risk_blocked=risk_blocked)
    if previous.get('retained_completion_attempt',0)>=1:
        pair.clear();pair.update(probe)
        if int(d['times'][i])>previous['retained_completion_deadline']:
            pair.update(closing=True,rehedge_failed=True)
            return dict(outcome,event='rehedge_unwind',outcome='unwind',reason='retained_completion_deadline_elapsed',retained_completion_attempt=1)
        if outcome['outcome']=='unfilled':
            pair.update(closing=True,rehedge_failed=True)
            return dict(outcome,event='rehedge_unwind',outcome='unwind',reason='retained_completion_zero_fill',retained_completion_attempt=1)
        return dict(outcome,retained_completion_attempt=1)
    partial_reasons={'partial_adjustment_hedge_mismatch','actual_new_hedge_outside_tolerance','partial_adjustment_gross_mismatch'}
    if outcome['outcome']=='unwind' and outcome['reason'] in partial_reasons and reduces_toward(previous['target'],previous['old_qty']):
        # Once any quantity changed, the cumulative owner PnL stop applies.
        checked=copy.deepcopy(pair);checked['has_rehedged']=True
        plan=completion_plan(checked,previous,ledger,i,d,cfg,p,sc,account_qty,equity,risk_blocked=risk_blocked)
        if plan['accepted']:
            pair.update(has_rehedged=True,rehedging=True,rehedge_plan=plan)
            return dict(time=int(d['times'][i]),pair=pair['pair'],event='retained_completion_pending',
                outcome='completion_pending',reason=plan['reason'],new_source=previous['new_source'],
                initial_adjustment_decision_time=previous['decision_time'],actual_qty=ledger.qty[pair['ids']].tolist(),
                original_target=previous['target'],remaining_delta=plan['delta'],completion_plan=plan)
        outcome=dict(outcome,retained_completion_rejection=plan['reason'])
    pair.clear();pair.update(probe)
    return outcome
