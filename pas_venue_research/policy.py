"""Isolated oracle-basis hypothesis with explicit executable costs.

This is a research proposal, not a calibrated return forecast or live policy.
No account, signer, exchange client, or order transmission is imported here.
"""
import math
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING


def price_tick(price,size_decimals):
    # Hyperliquid perps: <=5 significant figures and <=6-sizeDecimals decimals.
    significant=math.floor(math.log10(price))-4
    return Decimal(10)**max(significant,size_decimals-6)


def quote_price(value,size_decimals,is_buy):
    tick=price_tick(value,size_decimals)
    return float((Decimal(str(value))/tick).to_integral_value(
        rounding=ROUND_FLOOR if is_buy else ROUND_CEILING)*tick)


def walk(levels,units,fraction=.25):
    left=units;cash=0.
    for row in levels:
        price=float(row['px']);size=float(row['sz'])*fraction
        take=min(left,size);left-=take;cash+=take*price
        if left<=units*1e-12:return cash/units
    return None


def propose(sample,config,mode):
    """No return is credited from a stale last trade or a future book."""
    def reject(reason,**detail):
        return dict(accepted=False,reason=reason,mode=mode,**detail)
    if mode not in ('taker_basis','maker_basis'):raise ValueError('Unknown mode')
    if sample.get('gap') or not 0<=sample['now']-sample['book_received']<=config['book_max_age_seconds']:
        return reject('stale_book')
    if not 0<=sample['now']-sample['context_received']<=config['context_max_age_seconds']:
        return reject('stale_or_future_context')
    if not -1<=sample['now']-sample['book']['time']/1000<=config['book_max_age_seconds']:
        return reject('stale_or_future_server_book')
    oracle=float(sample['context']['oraclePx'])
    sigma=sample['reference_hourly_sigma']
    if not math.isfinite(oracle+sigma) or oracle<=0 or sigma<=0:
        return reject('missing_risk_input')
    levels=sample['book']['levels']
    if not all(levels):return reject('empty_book')
    bid=float(levels[0][0]['px']);ask=float(levels[1][0]['px'])
    if not math.isfinite(bid+ask) or bid<=0 or ask<=bid:return reject('invalid_book')
    mid=(bid+ask)/2
    side=1 if oracle>mid else -1
    decimals=sample['size_decimals']
    units=float((Decimal(str(config['notional']/mid))*Decimal(10)**decimals).to_integral_value(rounding=ROUND_FLOOR)/Decimal(10)**decimals)
    if units*mid<10:return reject('below_minimum')
    if mode=='taker_basis':
        entry=walk(levels[1 if side>0 else 0],units,config['depth_fraction'])
    else:
        entry=quote_price(bid if side>0 else ask,decimals,side>0)
    exit_proxy=walk(levels[0 if side>0 else 1],units,config['depth_fraction'])
    if entry is None or exit_proxy is None:return reject('insufficient_two_sided_depth')
    funding=float(sample['context']['funding'])
    hold=config['holding_seconds']
    # Explicit fixed hypothesis, not an assertion that a gap is an arbitrage.
    expected_terminal=mid+config['basis_convergence_fraction']*(oracle-mid)
    gross=side*(expected_terminal-entry)/mid*10000
    entry_fee=config['maker_fee_bps'] if mode=='maker_basis' else config['taker_fee_bps']
    fees=(entry_fee*entry+config['taker_fee_bps']*exit_proxy)/mid
    exit_cost=side*(mid-exit_proxy)/mid*10000+config['exit_impact_bps']
    adverse_funding=max(side*funding,0)*hold/3600*10000
    risk_reserve=config['risk_z']*sigma*math.sqrt(hold/3600)*10000
    required=fees+exit_cost+adverse_funding+risk_reserve+config['entry_buffer_bps']
    detail=dict(side=side,units=units,limit=quote_price(entry,decimals,side>0),
        entry_model_price=entry,notional=units*entry,
        oracle=oracle,mid=mid,oracle_mid_basis_bps=(mid/oracle-1)*10000,
        hypothesized_price_edge_bps=gross,required_edge_bps=required,
        entry_fee_bps=entry_fee,exit_fee_bps=config['taker_fee_bps'],
        risk_reserve_bps=risk_reserve,expected_terminal=expected_terminal,
        alpha_validated=False,live_eligible=False)
    if gross<=required:return reject('edge_below_cost_and_risk',**detail)
    return dict(accepted=True,reason='research_hypothesis_only',mode=mode,**detail)
