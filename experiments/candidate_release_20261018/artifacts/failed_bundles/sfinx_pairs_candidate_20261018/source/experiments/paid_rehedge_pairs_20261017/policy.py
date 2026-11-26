"""Costed inventory adjustment; all orders are offline replay intentions."""
import numpy as np
from experiments.inventory_renewal_pairs_20261017.policy import (
    held_state as prior_held_state, renewal_plan as prior_renewal_plan,
)
from replay_lab.common_comparison.engine import rounded, risk_breaches
from replay_lab.pullback.broker import execution_price


def anchor_stop(pair, ledger, prices):
    if 'stop_price_origin' in pair:
        return
    q = np.asarray(pair['risk_quantities'])
    entry = np.asarray(pair['risk_entry_prices'])
    current_unrealized = float(q @ (prices[pair['ids']] - entry))
    pair['stop_price_origin'] = ledger.equity(prices) + ledger.fees - ledger.funding - current_unrealized
    pair['stop_initial_gross'] = float(abs(q) @ entry)


def held_state(pair, i, d, de, cfg, *, ledger=None):
    if not pair.get('has_rehedged'):
        return prior_held_state(pair, i, d, de, cfg)
    if ledger is None or 'stop_price_origin' not in pair:
        raise ValueError('Adjusted inventory requires its actual episode ledger and original stop anchor')
    # The original quantity vector is no longer a valid PnL estimator.
    probe = dict(pair)
    probe.pop('risk_entry_prices', None)
    state = prior_held_state(probe, i, d, de, cfg)
    pnl = ledger.equity(d['par_close'][i]) + ledger.fees - ledger.funding - pair['stop_price_origin']
    value = pnl / pair['stop_initial_gross']
    if value <= -state['return_stop_bound'] and state['reason'] != 'hard_holding_limit':
        state['reason'] = 'pair_return_stop'
    return dict(state, pair_return=float(value), episode_price_pnl=float(pnl),
                stop_initial_gross=pair['stop_initial_gross'])


def exit_value(q, marks, terminal, ids, i, d, maturity, p, sc):
    fee = sc['fee_bps'] / 10000
    now = int(d['times'][i])
    close = np.array([execution_price(px, -np.sign(z), sc['impact_bps'], p['execution']['decimals'][j])
                      for j, z, px in zip(ids, q, marks)])
    future = np.array([execution_price(px, -np.sign(z), sc['impact_bps'], p['execution']['decimals'][j])
                       for j, z, px in zip(ids, q, terminal)])
    carry = -q * d['reference_close'][i, ids] * d['known_hourly_funding'][i, ids] * max(0., (maturity-now)/3600)
    carry = float(np.where(carry > 0, .5 * carry, carry).sum())
    exit_cost = float(q @ (terminal-future) + abs(q) @ future * fee)
    return dict(immediate_close_cash=float(q @ (close-marks) - abs(q) @ close * fee),
                expected_price_cash=float(q @ (terminal-marks)), expected_funding_cash=carry,
                estimated_future_exit_cost=exit_cost,
                expected_hold_cash=float(q @ (terminal-marks)) + carry-exit_cost)


def renewal_plan(pair, owned_qty, i, d, de, cfg, p, sc, account_qty, equity, *, risk_blocked=False, ledger=None):
    if not pair.get('paid_rehedge_enabled'):
        return prior_renewal_plan(pair, owned_qty, i, d, de, cfg, p, sc, account_qty, equity, risk_blocked=risk_blocked)
    state = held_state(pair, i, d, de, cfg, ledger=ledger)
    if state['reason'] is not None:
        return None
    probe = dict(pair)
    if pair.get('has_rehedged'):
        # The ledger-based stop above replaces the obsolete fixed-quantity stop.
        probe.pop('risk_entry_prices', None)
    old = prior_renewal_plan(probe, owned_qty, i, d, de, cfg, p, sc, account_qty, equity, risk_blocked=risk_blocked)
    if old is None or old['accepted'] or old['reason'] != 'new_hedge_requires_resizing':
        return old
    def reject(reason):
        return dict(old, reason=reason)
    now = int(d['times'][i]); ids = list(pair['ids']); k = pair['pair_index']
    q = np.asarray(owned_qty)[ids].copy(); marks = d['par_close'][i, ids]
    source = int(de['signal_times'][i, k]); maturity = source + pair['horizon'] * 3600
    if pair.get('last_rehedge_attempt_source', -1) >= source:
        return reject('adjustment_source_already_attempted')
    anchors = de['source_prices'][i, ids]; beta = float(de['beta'][i, k]); mu = float(de['mu'][i, k])
    terminal = anchors * np.array([1+mu, 1-mu])
    if np.any(terminal <= 0) or not np.isfinite(terminal).all():
        return reject('invalid_terminal_hypothesis')
    ratio = beta * anchors[0] / anchors[1]
    gross = min(float(abs(q) @ marks), cfg['pair_max_gross'])
    units = pair['direction'] * gross / (marks[0] + ratio * marks[1]) * np.array([1., -ratio])
    target = np.array([rounded(z, p['execution']['decimals'][j]) for j, z in zip(ids, units)])
    if np.any(target == 0):
        return reject('adjusted_quantity_rounds_to_zero')
    target_gross = float(abs(target) @ marks)
    if abs(target_gross/gross-1) > cfg['rehedge_policy']['gross_tolerance']:
        return reject('rounded_adjustment_does_not_preserve_gross')
    if abs(abs(target[1]/target[0])/ratio-1) > cfg['pair_balance_tolerance']:
        return reject('rounded_adjustment_misses_hedge')
    delta = target-q
    active = np.flatnonzero(abs(delta) > 1e-10)
    if not len(active):
        return reject('adjustment_has_no_trade')
    preview = marks.copy()
    for n in active:
        preview[n] = execution_price(marks[n], np.sign(delta[n]), sc['impact_bps'], p['execution']['decimals'][ids[n]])
        if abs(delta[n])*preview[n] < p['execution']['minimum_order_value']:
            return reject('adjustment_below_minimum_order')
    fee = sc['fee_bps']/10000
    adjustment_cost = float(delta @ (preview-marks) + abs(delta) @ preview * fee)
    value = exit_value(target, marks, terminal, ids, i, d, maturity, p, sc)
    immediate = exit_value(q, marks, terminal, ids, i, d, maturity, p, sc)['immediate_close_cash']
    advantage = value['expected_hold_cash'] - adjustment_cost - immediate
    buffer = gross * cfg['renewal_policy']['incremental_buffer_bps']/10000
    slack = advantage-buffer
    if not np.isfinite(slack) or slack <= 0:
        return reject('adjustment_cost_exceeds_holding_advantage')
    limits = preview.copy()
    for n in active:
        side = np.sign(delta[n])
        distance = slack/len(active)/(abs(delta[n])*(1+side*fee))
        # The economic limit is bounded away from negative prices on sells.
        distance = min(distance, preview[n] * .99)
        limits[n] += side * distance
    worst_cost = float(delta @ (limits-marks) + abs(delta) @ limits * fee)
    assert value['expected_hold_cash']-worst_cost-immediate >= buffer-1e-8
    for mask in ([1, 0], [0, 1], [1, 1]):
        risk_qty = np.asarray(account_qty).copy(); risk_qty[ids] += delta*np.array(mask)
        if risk_breaches(risk_qty, equity-worst_cost, d['par_close'][i], d['sigma'][i], p['risk']):
            return reject('adjustment_independent_leg_risk')
    return dict(accepted=True, requires_adjustment=True, reason='positive_costed_hedge_adjustment_value',
                old_maturity=old['old_maturity'], new_maturity=maturity, new_source=source,
                old_qty=q.tolist(), target=target.tolist(), delta=delta.tolist(), target_gross=target_gross,
                current_gross=float(abs(q)@marks), desired_unit_ratio=ratio,
                terminal_prices=terminal.tolist(), adjustment_cost_preview=adjustment_cost,
                immediate_close_cash=immediate, expected_hold_cash=value['expected_hold_cash'],
                expected_funding_cash=value['expected_funding_cash'],
                estimated_future_exit_cost=value['estimated_future_exit_cost'],
                expected_price_cash=value['expected_price_cash'], incremental_advantage=advantage,
                buffer_cash=buffer, entry_limits={str(ids[n]):float(limits[n]) for n in active},
                minimum_advantage_at_limits=value['expected_hold_cash']-worst_cost-immediate,
                hard_holding_deadline=state['hard_holding_deadline'],
                orders_required=len(active), entry_fees_charged_again=0, decision_time=now)


def reconcile_adjustment(pair, ledger, i, d, de, cfg, p, sc, account_qty, equity, *, all_outcomes_observed, risk_blocked=False):
    """Only actual fills can accept renewal; a failed partial adjustment exits."""
    if not all_outcomes_observed:
        # Absence of a received response is not evidence of a cancelled order.
        return dict(time=int(d['times'][i]), pair=pair['pair'], event='rehedge_outcome_unknown',
                    outcome='awaiting_reconciliation', reason='missing_order_outcome')
    plan = pair.pop('rehedge_plan'); pair['rehedging'] = False
    ids = pair['ids']; q = ledger.qty[ids].copy(); old = np.asarray(plan['old_qty'])
    now = int(d['times'][i])
    result = dict(time=now, pair=pair['pair'], new_source=plan['new_source'], old_maturity=plan['old_maturity'],
                  target=plan['target'], actual_qty=q.tolist(), actual_delta=(q-old).tolist())
    if np.all(abs(q-old) < 1e-10):
        return dict(result, event='rehedge_unfilled', outcome='unfilled', reason='no_pair_quantity_changed')
    pair['has_rehedged'] = True
    state = held_state(pair, i, d, de, cfg, ledger=ledger)
    def unwind(reason):
        pair.update(closing=True, rehedge_failed=True)
        return dict(result, event='rehedge_unwind', outcome='unwind', reason=reason)
    if state['reason'] is not None:
        return unwind(state['reason'])
    if risk_blocked or risk_breaches(account_qty, equity, d['par_close'][i], d['sigma'][i], p['risk']):
        return unwind('arrival_global_risk_or_halt')
    target = np.asarray(plan['target']); ratios = q/target
    if min(ratios) <= 0 or abs(ratios[0]-ratios[1])/max(ratios) > cfg['pair_balance_tolerance']:
        return unwind('partial_adjustment_hedge_mismatch')
    if abs(abs(q[1]/q[0])/plan['desired_unit_ratio']-1) > cfg['pair_balance_tolerance']:
        return unwind('actual_new_hedge_outside_tolerance')
    marks = d['par_close'][i, ids]; gross = float(abs(q)@marks)
    if abs(gross/(abs(target)@marks)-1) > cfg['rehedge_policy']['gross_tolerance']:
        return unwind('partial_adjustment_gross_mismatch')
    value = exit_value(q, marks, np.array(plan['terminal_prices']), ids, i, d, plan['new_maturity'], p, sc)
    advantage = value['expected_hold_cash']-value['immediate_close_cash']
    buffer = gross*cfg['renewal_policy']['incremental_buffer_bps']/10000
    if plan['new_maturity'] <= now or not np.isfinite(advantage) or advantage <= buffer:
        return unwind('arrival_remaining_value_insufficient')
    pair.update(target=q.tolist(), renewal_maturity=plan['new_maturity'], latest_renewal_source=plan['new_source'])
    return dict(result, event='renewed', outcome='renewed', reason='actual_hedge_adjustment_reconciled',
                new_maturity=plan['new_maturity'], hard_holding_deadline=plan['hard_holding_deadline'],
                orders_required=plan['orders_required'], entry_fees_charged_again=0,
                incremental_advantage=advantage, buffer_cash=buffer, adjustment_decision_time=plan['decision_time'])
