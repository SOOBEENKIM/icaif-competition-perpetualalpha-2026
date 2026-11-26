"""PAS entry-cost and size eligibility, independent of a trading client.

Weights are allocation instructions, not expected returns. A missing calibrated
PAS return estimate is explicitly ineligible for capital expansion.
"""
import math


def average_price(levels,units,depth_fraction=1.):
    if units<=0 or not 0<depth_fraction<=1:return None
    left=units;value=0.
    for level in levels:
        price=float(level['px']);size=float(level['sz'])*depth_fraction
        if not math.isfinite(price+size) or price<=0 or size<0:return None
        take=min(left,size);value+=take*price;left-=take
        if left<=units*1e-12:return value/units
    return None


def evaluate_size(book,notional,side,*,fee_bps,depth_fraction,expected_pas_return_bps=None,
                  funding_rate=0.,hours=24,exit_impact_bps=10.,buffer_bps=5.,
                  forecast_validated=False):
    if (side not in (-1,1) or not math.isfinite(notional) or notional<=0
            or not all(math.isfinite(x) for x in (fee_bps,depth_fraction,funding_rate,hours,exit_impact_bps,buffer_bps))
            or min(fee_bps,hours,exit_impact_bps,buffer_bps)<0 or not 0<depth_fraction<=1):
        raise ValueError('Invalid cost/size inputs')
    levels=book['levels']
    if not all(levels):return dict(eligible=False,reason='missing_depth')
    bid=float(levels[0][0]['px']);ask=float(levels[1][0]['px'])
    if not math.isfinite(bid+ask) or bid<=0 or ask<bid:
        return dict(eligible=False,reason='invalid_book')
    mid=(bid+ask)/2;units=notional/mid
    entry=average_price(levels[1 if side>0 else 0],units,depth_fraction)
    exit_=average_price(levels[0 if side>0 else 1],units,depth_fraction)
    if entry is None or exit_ is None:return dict(eligible=False,reason='insufficient_visible_depth')
    spread_impact=side*(entry-exit_)/mid*10000
    fees=fee_bps*(entry+exit_)/mid
    # Do not assume today's favorable rate will finance tomorrow's trade.
    adverse_funding=max(side*funding_rate,0.)*hours*10000
    cost=spread_impact+fees+adverse_funding+exit_impact_bps
    evidence=dict(notional=notional,side=side,roundtrip_book_and_fees_bps=spread_impact+fees,
        adverse_funding_scenario_bps=adverse_funding,exit_impact_reserve_bps=exit_impact_bps,
        total_cost_reserve_bps=cost,required_price_edge_bps=cost+buffer_bps,
        funding_horizon_hours=hours,future_exit_liquidity_guaranteed=False)
    if (expected_pas_return_bps is None or not math.isfinite(expected_pas_return_bps)
            or not forecast_validated):
        return dict(evidence,eligible=False,reason='no_validated_pas_alpha',expected_net_bps=None)
    net=side*expected_pas_return_bps-cost
    return dict(evidence,expected_net_bps=net,eligible=net>buffer_bps,
        reason='positive_net_edge' if net>buffer_bps else 'edge_below_full_cost')


def maximum_eligible_size(rows):
    eligible=[r['notional'] for r in rows if r.get('eligible')]
    return max(eligible) if eligible else 0.
