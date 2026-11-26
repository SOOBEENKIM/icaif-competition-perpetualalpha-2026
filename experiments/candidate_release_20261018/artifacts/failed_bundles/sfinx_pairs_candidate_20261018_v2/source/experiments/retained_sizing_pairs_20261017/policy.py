"""Costed retained inventory choices after a valid original renewal decision."""
import numpy as np
from experiments.paid_rehedge_pairs_20261017.policy import (
    anchor_stop, held_state, exit_value,
    renewal_plan as original_plan, reconcile_adjustment as original_reconcile,
)
from replay_lab.common_comparison.engine import rounded, risk_breaches
from replay_lab.pullback.broker import execution_price


def variance_charge(target, owned, account_qty, ids, marks, sigma, equity, hours, gamma):
    if equity <= 0 or hours <= 0 or gamma < 0:
        raise ValueError('Invalid utility horizon, equity or risk aversion')
    baseline=np.asarray(account_qty, dtype=float)*marks
    baseline=baseline.copy();baseline[ids]-=np.asarray(owned)*marks[ids]
    portfolio=baseline.copy();portfolio[ids]+=np.asarray(target)*marks[ids]
    # The stored covariance is per hour; annual/daily multipliers are not used.
    delta=float(portfolio@sigma@portfolio-baseline@sigma@baseline)
    return gamma*hours*delta/(2*equity)


def value_at(target, owned, account_qty, ids, i, d, maturity, terminal, p, sc, gamma, equity):
    target=np.asarray(target);owned=np.asarray(owned)
    marks=d['par_close'][i];delta=target-owned
    preview=np.array([execution_price(marks[j],np.sign(z),sc['impact_bps'],p['execution']['decimals'][j])
                      if abs(z)>1e-10 else marks[j] for j,z in zip(ids,delta)])
    fee=sc['fee_bps']/10000
    cost=float(delta@(preview-marks[ids])+abs(delta)@preview*fee)
    future=exit_value(target,marks[ids],terminal,ids,i,d,maturity,p,sc)
    hours=(maturity-int(d['times'][i]))/3600
    risk=variance_charge(target,owned,account_qty,ids,marks,d['sigma'][i],equity,hours,gamma)
    return dict(target=target.tolist(),delta=delta.tolist(),preview=preview.tolist(),
                adjustment_cost_preview=cost,risk_charge=risk,utility=future['expected_hold_cash']-cost-risk,
                **future)


def renewal_plan(pair,owned_qty,i,d,de,cfg,p,sc,account_qty,equity,*,risk_blocked=False,ledger=None):
    original=original_plan(pair,owned_qty,i,d,de,cfg,p,sc,account_qty,equity,risk_blocked=risk_blocked,ledger=ledger)
    if not pair.get('utility_renewal_enabled') or original is None or not original['accepted']:
        return original
    policy=cfg['retained_sizing'];gamma=policy['risk_aversion'];ids=list(pair['ids'])
    owned=np.asarray(owned_qty)[ids].copy();marks=d['par_close'][i];now=int(d['times'][i])
    maturity=original['new_maturity'];terminal=np.array(original['terminal_prices'])
    default=np.array(original['target'] if original.get('requires_adjustment') else owned)
    gross=float(abs(owned)@marks[ids]);default_gross=float(abs(default)@marks[ids])
    limit=min(pair['stop_initial_gross'],cfg['pair_max_gross'])
    reference=value_at(default,owned,account_qty,ids,i,d,maturity,terminal,p,sc,gamma,equity)
    threshold=gross*policy['utility_buffer_bps']/10000
    immediate=exit_value(owned,marks[ids],terminal,ids,i,d,maturity,p,sc)['immediate_close_cash']
    choices=[]
    for fraction in [0.,*policy['gross_fractions']]:
        target=np.array([rounded(q*fraction*limit/default_gross,p['execution']['decimals'][j]) for j,q in zip(ids,default)])
        if np.array_equal(target,default):continue
        flat=not np.any(target)
        if not flat:
            if np.any(abs(target)*marks[ids]<p['execution']['minimum_order_value']):continue
            ratio=abs(target[1]/target[0]);desired=original['desired_unit_ratio']
            if abs(ratio/desired-1)>cfg['pair_balance_tolerance']:continue
        value=value_at(target,owned,account_qty,ids,i,d,maturity,terminal,p,sc,gamma,equity)
        active=np.flatnonzero(abs(target-owned)>1e-10)
        if not len(active):continue
        preview=np.array(value['preview']);delta=target-owned
        if any(abs(delta[n])*preview[n]<p['execution']['minimum_order_value'] for n in active):continue
        for mask in ([1,0],[0,1],[1,1]):
            check=np.asarray(account_qty).copy();check[ids]+=delta*np.array(mask)
            if risk_breaches(check,equity-value['adjustment_cost_preview'],marks,d['sigma'][i],p['risk']):break
        else:
            advantage=value['utility']-reference['utility']
            cash_advantage=value['expected_hold_cash']-value['adjustment_cost_preview']-immediate
            retained_gross=float(abs(target)@marks[ids]);cash_buffer=retained_gross*cfg['renewal_policy']['incremental_buffer_bps']/10000
            if advantage>threshold and (flat or cash_advantage>cash_buffer):
                choices.append(dict(value,flat=flat,utility_advantage=advantage,cash_advantage=cash_advantage,
                                    cash_buffer=cash_buffer,target_gross=retained_gross))
    if not choices:return original
    chosen=max(choices,key=lambda x:(x['utility'],-sum(abs(z) for z in x['delta'])))
    common=dict(utility_enabled=True,risk_aversion=gamma,original_renewal_utility=reference['utility'],
                retained_utility=chosen['utility'],utility_advantage=chosen['utility_advantage'],
                utility_buffer_cash=threshold,risk_charge=chosen['risk_charge'],
                original_balanced_gross=pair['stop_initial_gross'],current_gross=gross)
    if chosen['flat']:
        return dict(accepted=False,close_for_utility=True,reason='risk_adjusted_paid_close_preferred',
                    old_maturity=original['old_maturity'],new_source=original['new_source'],
                    immediate_close_cash=immediate,orders_required=len(np.flatnonzero(owned)),**common)
    # Do not allow the sealed price slack to consume either the utility gain
    # or the already-costed retained-position continuation value.
    slack=min(chosen['utility_advantage']-threshold,chosen['cash_advantage']-chosen['cash_buffer'])
    target=np.array(chosen['target']);delta=target-owned;active=np.flatnonzero(abs(delta)>1e-10)
    limits=np.array(chosen['preview']);fee=sc['fee_bps']/10000
    for n in active:
        side=np.sign(delta[n]);distance=slack/len(active)/(abs(delta[n])*(1+side*fee))
        limits[n]+=side*min(distance,limits[n]*.99)
    worst=float(delta@(limits-marks[ids])+abs(delta)@limits*fee)
    minimum_utility=chosen['expected_hold_cash']-worst-chosen['risk_charge']-reference['utility']
    minimum_cash=chosen['expected_hold_cash']-worst-immediate
    assert minimum_utility>=threshold-1e-8
    assert minimum_cash>=chosen['cash_buffer']-1e-8
    for mask in ([1,0],[0,1],[1,1]):
        check=np.asarray(account_qty).copy();check[ids]+=delta*np.array(mask)
        if risk_breaches(check,equity-worst,marks,d['sigma'][i],p['risk']):return original
    return dict(accepted=True,requires_adjustment=True,reason='costed_retained_size_utility_improved',
        old_maturity=original['old_maturity'],new_maturity=maturity,new_source=original['new_source'],
        old_qty=owned.tolist(),target=target.tolist(),delta=delta.tolist(),target_gross=chosen['target_gross'],
        desired_unit_ratio=original['desired_unit_ratio'],terminal_prices=terminal.tolist(),
        adjustment_cost_preview=chosen['adjustment_cost_preview'],immediate_close_cash=immediate,
        expected_hold_cash=chosen['expected_hold_cash'],expected_funding_cash=chosen['expected_funding_cash'],
        estimated_future_exit_cost=chosen['estimated_future_exit_cost'],expected_price_cash=chosen['expected_price_cash'],
        incremental_advantage=chosen['cash_advantage'],buffer_cash=chosen['cash_buffer'],
        entry_limits={str(ids[n]):float(limits[n]) for n in active},minimum_advantage_at_limits=minimum_cash,
        minimum_utility_advantage_at_limits=minimum_utility,hard_holding_deadline=original['hard_holding_deadline'],
        orders_required=len(active),entry_fees_charged_again=0,decision_time=now,**common)


def reconcile_adjustment(pair,ledger,i,d,de,cfg,p,sc,account_qty,equity,*,all_outcomes_observed,risk_blocked=False):
    plan=pair.get('rehedge_plan',{})
    outcome=original_reconcile(pair,ledger,i,d,de,cfg,p,sc,account_qty,equity,
        all_outcomes_observed=all_outcomes_observed,risk_blocked=risk_blocked)
    if not plan.get('utility_enabled') or outcome['outcome']!='renewed':return outcome
    q=ledger.qty[pair['ids']].copy();terminal=np.array(plan['terminal_prices'])
    value=value_at(q,q,account_qty,pair['ids'],i,d,plan['new_maturity'],terminal,p,sc,plan['risk_aversion'],equity)
    # Adjustments already filled are sunk. Only the remaining position risk and
    # paid alternatives are assessed; original reconciliation accounted for fills.
    advantage=value['utility']-value['immediate_close_cash']
    buffer=float(abs(q)@d['par_close'][i,pair['ids']])*cfg['renewal_policy']['incremental_buffer_bps']/10000
    if not np.isfinite(advantage) or advantage<=buffer:
        pair.update(closing=True,rehedge_failed=True)
        return dict(outcome,event='rehedge_unwind',outcome='unwind',reason='arrival_retained_utility_insufficient',
                    arrival_utility_advantage=advantage,arrival_utility_buffer=buffer)
    return dict(outcome,arrival_utility_advantage=advantage,arrival_utility_buffer=buffer)
