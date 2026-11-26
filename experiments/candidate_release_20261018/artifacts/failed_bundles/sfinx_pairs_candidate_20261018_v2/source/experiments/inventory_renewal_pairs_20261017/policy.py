"""Renew an existing paired holding from a newly available price forecast."""
import numpy as np
from experiments.dense_history_pairs_20261017.policy import held_state as fixed_held_state
from replay_lab.common_comparison.engine import risk_breaches
from replay_lab.pullback.broker import execution_price


def held_state(pair,i,d,de,cfg):
    state=fixed_held_state(pair,i,d,de,cfg)
    if not pair.get('renewable'):return state
    now=int(d['times'][i]);step=int(d['times'][1]-d['times'][0])
    original=pair['forecast_source']+pair['horizon']*3600
    hard=pair.get('first_fill',pair['forecast_source'])+cfg['renewal_policy']['maximum_holding_hours']*3600
    maturity=min(pair.get('renewal_maturity',original),hard)
    reason=None
    if now>=hard-step:reason='hard_holding_limit'
    elif state['pair_return'] is not None and state['pair_return']<=-state['return_stop_bound']:reason='pair_return_stop'
    elif now>=maturity-step:reason='forecast_maturity'
    return dict(state,reason=reason,maturity=maturity,original_maturity=original,hard_holding_deadline=hard)


def renewal_plan(pair,owned_qty,i,d,de,cfg,p,sc,account_qty,equity,*,risk_blocked=False):
    now=int(d['times'][i]);policy=cfg['renewal_policy'];cadence=policy['decision_cadence_hours']*3600
    step=int(d['times'][1]-d['times'][0]);old_maturity=pair.get('renewal_maturity',pair['forecast_source']+pair['horizon']*3600)
    if now%cadence or not old_maturity-cadence<=now<old_maturity-step:return None
    def reject(reason):return dict(accepted=False,reason=reason,old_maturity=old_maturity)
    if not pair.get('renewable') or not pair.get('balanced') or any(pair.get(k,False) for k in ['closing','completing','repairing']):
        return reject('inventory_not_renewable')
    state=held_state(pair,i,d,de,cfg)
    if state['reason'] is not None:return reject('existing_exit_or_stop')
    prices=d['par_close'][i];sigma=d['sigma'][i]
    if risk_blocked or risk_breaches(account_qty,equity,prices,sigma,p['risk']):return reject('global_risk_or_halt')
    ids=list(pair['ids']);k=pair['pair_index']
    if list(de['pairs'][k])!=ids or cfg['dense_pairs'][k]!=pair['pair']:raise ValueError('Renewal pair identity mismatch')
    source=int(de['signal_times'][i,k]);latest=pair.get('latest_renewal_source',pair['forecast_source'])
    if not de['directions'][i,k] or not latest<source<=now or now-source>=cfg['dense_policy']['source_lifetime_hours']*3600:
        return reject('no_new_available_signal')
    maturity=source+pair['horizon']*3600
    if maturity<=old_maturity:return reject('forecast_does_not_extend_holding')
    if maturity>state['hard_holding_deadline']:return reject('forecast_exceeds_hard_holding_limit')
    q=np.asarray(owned_qty)[ids].copy();ratios=q/np.asarray(pair['target'])
    if min(ratios)<=0 or abs(ratios[0]-ratios[1])/max(ratios)>cfg['pair_balance_tolerance']:
        return reject('current_inventory_unbalanced')
    if np.sign(q[0])!=pair['direction'] or np.sign(q[1])!=-pair['direction']:return reject('position_direction_changed')
    anchors=de['source_prices'][i,ids];beta=float(de['beta'][i,k]);mu=float(de['mu'][i,k])
    if not np.isfinite([beta,mu,*anchors]).all() or beta<=0 or np.any(anchors<=0):return reject('unavailable_new_forecast')
    desired_unit_ratio=beta*anchors[0]/anchors[1];actual_ratio=abs(q[1]/q[0])
    hedge_error=abs(actual_ratio/desired_unit_ratio-1)
    if hedge_error>cfg['pair_balance_tolerance']:return reject('new_hedge_requires_resizing')
    terminal=anchors*np.array([1+mu,1-mu])
    if np.any(terminal<=0):return reject('invalid_terminal_hypothesis')
    close_prices=np.array([execution_price(prices[j],-np.sign(z),sc['impact_bps'],p['execution']['decimals'][j]) for j,z in zip(ids,q)])
    future_exit=np.array([execution_price(px,-np.sign(z),sc['impact_bps'],p['execution']['decimals'][j]) for j,z,px in zip(ids,q,terminal)])
    fee=sc['fee_bps']/10000;gross=float(abs(q)@prices[ids])
    immediate_close_cash=float(q@(close_prices-prices[ids])-abs(q)@close_prices*fee)
    expected_price_cash=float(q@(terminal-prices[ids]));remaining=(maturity-now)/3600
    carry=-q*d['reference_close'][i,ids]*d['known_hourly_funding'][i,ids]*remaining
    carry=float(np.where(carry>0,.5*carry,carry).sum())
    exit_cost=float(q@(terminal-future_exit)+abs(q)@future_exit*fee)
    hold_cash=expected_price_cash+carry-exit_cost;advantage=hold_cash-immediate_close_cash
    buffer=gross*policy['incremental_buffer_bps']/10000
    if not np.isfinite(advantage) or advantage<=buffer:return reject('holding_value_below_incremental_cost_buffer')
    assert source<=now<old_maturity and maturity<=state['hard_holding_deadline']
    return dict(accepted=True,reason='positive_incremental_holding_value',old_maturity=old_maturity,new_maturity=maturity,
        new_source=source,owned_qty=q.tolist(),desired_unit_ratio=desired_unit_ratio,actual_unit_ratio=actual_ratio,
        hedge_ratio_error=hedge_error,source_price_alpha=mu,terminal_prices=terminal.tolist(),
        expected_price_cash=expected_price_cash,expected_funding_cash=carry,estimated_future_exit_cost=exit_cost,
        immediate_close_cash=immediate_close_cash,expected_hold_cash=hold_cash,incremental_advantage=advantage,
        buffer_cash=buffer,orders_required=0,entry_fees_charged_again=0,hard_holding_deadline=state['hard_holding_deadline'])
