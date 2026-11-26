"""Incremental paid-close comparisons; never spend entry decision scores twice."""
import numpy as np
from experiments.paid_rehedge_pairs_20261017.policy import (
    held_state, anchor_stop, exit_value,
    renewal_plan as original_renewal,
    reconcile_adjustment as original_reconcile,
)
from .model import terminal_for_maturity


def remaining_check(pair, owned_qty, i, d, de, cfg, p, sc, state):
    policy = cfg.get('remaining_value', {})
    now = int(d['times'][i])
    if policy.get('mode') != 'remaining' or state['reason'] is not None:
        return None
    if now % (policy['decision_cadence_hours']*3600):
        return None
    if not pair.get('balanced') or any(pair.get(k, False) for k in ('closing','completing','repairing','rehedging')):
        return None
    forecast = terminal_for_maturity(de, i, pair['pair_index'], now, state['maturity'])
    if forecast is None:
        return dict(close=False, reason='remaining_model_unavailable', available=False)
    ids = pair['ids']; q = np.asarray(owned_qty)[ids]; marks = d['par_close'][i, ids]
    value = exit_value(q, marks, forecast['terminal_prices'], ids, i, d, state['maturity'], p, sc)
    advantage = value['expected_hold_cash']-value['immediate_close_cash']
    buffer = float(abs(q)@marks)*policy['exit_buffer_bps']/10000
    return dict(close=bool(advantage < -buffer), reason='negative_remaining_cash' if advantage < -buffer else 'retain_current_inventory',
                available=True, source=forecast['source'], horizon_hours=forecast['horizon_hours'],
                fit_time=forecast['fit_time'], terminal_prices=forecast['terminal_prices'].tolist(),
                holding_advantage=advantage, exit_buffer_cash=buffer, owned_qty=q.tolist(),
                entry_fees_charged_again=0, **value)


def renewal_plan(pair, owned_qty, i, d, de, cfg, p, sc, account_qty, equity, *, risk_blocked=False, ledger=None):
    original = original_renewal(pair, owned_qty, i, d, de, cfg, p, sc, account_qty, equity,
                                risk_blocked=risk_blocked, ledger=ledger)
    if not cfg.get('remaining_value', {}).get('mode') or original is None or not original['accepted']:
        return original
    now = int(d['times'][i]); ids = pair['ids']
    forecast = terminal_for_maturity(de, i, pair['pair_index'], now, original['new_maturity'])
    if forecast is None:
        return dict(original, remaining_model_available=False)
    owned = np.asarray(owned_qty)[ids]; marks = d['par_close'][i, ids]
    target = np.asarray(original['target']) if original.get('requires_adjustment') else owned
    value = exit_value(target, marks, forecast['terminal_prices'], ids, i, d, original['new_maturity'], p, sc)
    immediate = exit_value(owned, marks, forecast['terminal_prices'], ids, i, d, original['new_maturity'], p, sc)['immediate_close_cash']
    cost = 0.
    if original.get('requires_adjustment'):
        delta = target-owned
        limits = np.array([original['entry_limits'].get(str(j), marks[n]) for n,j in enumerate(ids)])
        cost = float(delta@(limits-marks)+abs(delta)@limits*sc['fee_bps']/10000)
    advantage = value['expected_hold_cash']-cost-immediate
    info = dict(remaining_model_available=True, joint_terminal_prices=forecast['terminal_prices'].tolist(),
                joint_horizon_hours=forecast['horizon_hours'], joint_fit_time=forecast['fit_time'],
                joint_source=forecast['source'], joint_advantage_at_sealed_limits=advantage,
                joint_worst_adjustment_cost=cost, joint_hold_cash=value['expected_hold_cash'])
    if not np.isfinite(advantage) or advantage <= original['buffer_cash']:
        return dict(accepted=False, reason='remaining_model_renewal_veto', old_maturity=original['old_maturity'],
                    proposed_maturity=original['new_maturity'], buffer_cash=original['buffer_cash'], **info)
    return dict(original, **info)


def reconcile_adjustment(pair, ledger, i, d, de, cfg, p, sc, account_qty, equity, *, all_outcomes_observed, risk_blocked=False):
    plan = pair.get('rehedge_plan', {}).copy()
    result = original_reconcile(pair, ledger, i, d, de, cfg, p, sc, account_qty, equity,
                                all_outcomes_observed=all_outcomes_observed, risk_blocked=risk_blocked)
    if result['outcome'] != 'renewed' or 'joint_terminal_prices' not in plan:
        return result
    ids = pair['ids']; q = ledger.qty[ids]; marks = d['par_close'][i, ids]
    value = exit_value(q, marks, np.array(plan['joint_terminal_prices']), ids, i, d, plan['new_maturity'], p, sc)
    advantage = value['expected_hold_cash']-value['immediate_close_cash']
    buffer = float(abs(q)@marks)*cfg['renewal_policy']['incremental_buffer_bps']/10000
    if not np.isfinite(advantage) or advantage <= buffer:
        pair.update(closing=True, rehedge_failed=True)
        return dict(result, event='rehedge_unwind', outcome='unwind', reason='arrival_joint_remaining_cash_insufficient',
                    joint_arrival_advantage=advantage, joint_arrival_buffer=buffer)
    return dict(result, joint_arrival_advantage=advantage, joint_arrival_buffer=buffer)
