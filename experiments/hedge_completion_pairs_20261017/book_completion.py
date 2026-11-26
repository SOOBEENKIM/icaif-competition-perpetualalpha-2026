"""One extra observed-book IOC attempt after a known unfilled hedge."""
import numpy as np
from experiments.basis_carry_20261017.probe import asof,delayed_snapshot,sweep,walk
from experiments.quote_anchored_pairs_20261017.pricing import reservation_limits


def plan(probe,fill,at,requested,streams,decimals,cfg,sc,d,symbols):
    pair=probe['pair'];lead=probe['lead'];other=probe['hedge'];li=pair.index(lead);hi=1-li
    snaps=[asof(streams[s],at,cfg) if s in streams else None for s in pair]
    if any(s is None for s in snaps):return None,'missing_completion_context'
    qty=np.zeros(2);qty[li]=fill['side']*fill['qty'];qty[hi]=-fill['side']*requested
    anchor=np.asarray(probe['anchor_prices']);side=int(np.sign(qty[hi]))
    preview=walk(snaps[hi]['book']['levels'][1 if side>0 else 0],requested,sc['depth_fraction'])
    if preview is None:return None,'insufficient_completion_depth'
    prices=np.zeros(2);prices[li]=fill['price'];prices[hi]=preview
    if requested*preview<cfg['minimum_hedge_notional']:return None,'completion_below_minimum'
    remaining=(probe['forecast_maturity_ms']-at)/3600000
    if remaining<=0:return None,'forecast_expired'
    rates=np.asarray([float(s['ctx']['funding']) for s in snaps]);oracle=np.asarray([s['oracle'] for s in snaps])
    carry=-qty*oracle*rates*remaining;carry=float(np.where(carry>0,.5*carry,carry).sum())
    accrued=0.;j=symbols.index(lead)
    settled=(d['times']%3600==0)&(d['times']*1000>fill['time'])&(d['times']*1000<=at)
    for row in np.flatnonzero(settled):
        amount=-qty[li]*float(d['reference_close'][row,j])*float(d['funding'][row,j])
        accrued+=amount*sc['funding_receipt_multiplier'] if amount>0 else amount
    carry+=accrued
    fees=np.full(2,sc['taker_fee_bps']);fees[li]=sc['maker_fee_bps']
    slack=np.zeros(2);slack[hi]=1
    budget=reservation_limits(qty,anchor,prices,probe['direction']*probe['forecast'],carry,
        sc['taker_fee_bps'],sc['exit_impact_bps'],cfg['completion_buffer_bps'],10000,
        entry_fee_bps=fees,slack_weights=slack)
    if budget is None:return None,'completion_edge_exhausted'
    limit=budget['limits'][hi]
    if limit<=0:return None,'invalid_completion_price'
    return dict(decision_time=at,hedge=other,requested=requested,side=side,limit=limit,
        expected_net_cash=budget['expected_net_cash'],minimum_expected_cash_at_limits=budget['minimum_expected_cash_at_limits'],
        buffer_cash=budget['buffer_cash'],accrued_lead_funding=accrued,
        frozen_forecast=probe['forecast'],frozen_anchor=anchor.tolist()),'planned'


def attempt(probe,fill,initial_hedge_time,requested,streams,decimals,cfg,sc,d,symbols):
    # The first observed IOC outcome must be known before a second is planned.
    at=initial_hedge_time+cfg['completion_ack_delay_ms']
    budget,status=plan(probe,fill,at,requested,streams,decimals,cfg,sc,d,symbols)
    record=dict(decision_time=at,status=status,got=0.,price=None,known_outcome=True)
    if budget is None:return record
    record.update(budget)
    arrival=delayed_snapshot(streams[probe['hedge']],at+cfg['entry_latency_ms'],cfg)
    if arrival is None:
        return dict(record,status='missing_completion_arrival',known_outcome=False,
            end_time=at+cfg['entry_latency_ms']+cfg['future_exit_observation_grace_ms'])
    end,snap=arrival
    got,price=sweep(snap['book']['levels'][1 if budget['side']>0 else 0],requested,
        sc['depth_fraction'],budget['side'],decimals[probe['hedge']],budget['limit'])
    status='full' if got>=requested-1e-10 else 'partial' if got else 'unfilled'
    return dict(record,status=status,got=got,price=price,end_time=end)
