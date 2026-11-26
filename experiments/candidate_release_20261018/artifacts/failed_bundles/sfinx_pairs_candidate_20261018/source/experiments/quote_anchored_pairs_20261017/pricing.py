"""Forecast-consistent cash budgets sealed into independent leg limits."""
import numpy as np
from replay_lab.common_comparison.engine import execute_intent as original_execute_intent
from replay_lab.pullback.broker import execution_price


def value_at_entry(qty, anchors, entry_prices, expected_return, expected_carry, fee_bps, exit_impact_bps, *, entry_fee_bps=None):
    qty=np.asarray(qty,float);anchors=np.asarray(anchors,float);entry_prices=np.asarray(entry_prices,float)
    gross=float(abs(qty)@anchors)
    expected_price=gross*expected_return-float(qty@(entry_prices-anchors))
    rates=np.full(len(qty),fee_bps) if entry_fee_bps is None else np.asarray(entry_fee_bps,float)
    entry_fee=float(np.sum(abs(qty)*entry_prices*rates))/10000
    exit_reserve=gross*(fee_bps+exit_impact_bps)/10000
    return dict(expected_net_cash=expected_price+expected_carry-entry_fee-exit_reserve,
        expected_price_cash=expected_price,expected_funding_cash=expected_carry,
        entry_fee_cash=entry_fee,exit_reserve_cash=exit_reserve,anchor_gross=gross)


def reservation_limits(qty, anchors, previews, expected_return, expected_carry, fee_bps, exit_impact_bps, buffer_bps, cap_bps=10, *, entry_fee_bps=None, slack_weights=None):
    qty=np.asarray(qty,float);anchors=np.asarray(anchors,float);previews=np.asarray(previews,float)
    if not len(qty) or np.any(qty==0):return None
    values=value_at_entry(qty,anchors,previews,expected_return,expected_carry,fee_bps,exit_impact_bps,entry_fee_bps=entry_fee_bps)
    gross=values['anchor_gross'];slack=values['expected_net_cash']-gross*buffer_bps/10000
    if slack<=0:return None
    # For an adverse signed price movement, the incremental entry price and
    # fee cost is |q|*distance*(1+sign(q)*fee_rate).
    signs=np.sign(qty);weights=abs(qty)*anchors/gross if slack_weights is None else np.asarray(slack_weights,float)
    assert np.all(weights>=0) and abs(weights.sum()-1)<1e-10
    rates=np.full(len(qty),fee_bps) if entry_fee_bps is None else np.asarray(entry_fee_bps,float)
    distance=slack*weights/(abs(qty)*(1+signs*rates/10000))
    distance=np.minimum(distance,previews*cap_bps/10000)
    limits=previews+signs*distance
    worst=value_at_entry(qty,anchors,limits,expected_return,expected_carry,fee_bps,exit_impact_bps,entry_fee_bps=entry_fee_bps)
    assert worst['expected_net_cash']>=gross*buffer_bps/10000-1e-8
    return dict(**values,limits=limits.tolist(),minimum_expected_cash_at_limits=worst['expected_net_cash'],
        buffer_cash=gross*buffer_bps/10000)


def execute_intent(ledger,intent,market,index,protocol,scenario):
    # Never inspect one leg's future price to cancel the other leg. Each
    # order uses its own reservation limit already fixed at decision time.
    guarded=dict(intent);guarded['source_volume']=intent['source_volume'].copy()
    blocked=[];limits=intent.get('entry_limits',{})
    for j,(old,target) in enumerate(zip(ledger.qty,intent['target_qty'])):
        delta=target-old;reducing=old!=0 and (target*old<=0 or abs(target)<abs(old))
        if not delta or reducing:continue
        sign=np.sign(delta);decimals=protocol['execution']['decimals'][j]
        preview=execution_price(intent['source_prices'][j],sign,scenario['impact_bps'],decimals)
        cap=preview*(1+sign*.001)
        limit=limits.get(str(j),cap)
        price=execution_price(market.close[index,j],sign,scenario['impact_bps'],decimals)
        if sign*(price-limit)>1e-10:
            guarded['source_volume'][j]=0;blocked.append(market.symbols[j])
    orders=original_execute_intent(ledger,guarded,market,index,protocol,scenario)
    for o in orders:o['entry_limit_blocked']=o['symbol'] in blocked
    return orders
