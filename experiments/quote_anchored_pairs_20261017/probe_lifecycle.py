"""Quote-selected probes with frozen hedge units/limit and source-anchored maturity."""
import copy
import json
import time
from collections import Counter
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape, asof, delayed_snapshot, sweep
from experiments.execution_retry_20261016.engine import lot_floor
from experiments.complementary_alpha_20261017.research import protected
from experiments.par_direct_forecast_20261017.data import sha, save
from strategy.agent import TAIL
from experiments.pair_execution_calibration_20261017.research import HERE, ROOT, OUT, HISTORY


def apply_fill(position, at, signed_qty, price, fee_bps, source):
    if not signed_qty:
        return
    # Keep balances on the exchange's size grid. Float subtraction such as
    # .0015 - .0014 is slightly below .0001 and would otherwise be floored to
    # zero by a subsequent reduce-only order, stranding a whole lot forever.
    units=round(signed_qty*10**position['decimals'])
    assert abs(units/10**position['decimals']-signed_qty)<1e-9
    position['units'] += units
    position['qty'] = position['units']/10**position['decimals']
    position['cash'] -= signed_qty * price
    fee = abs(signed_qty) * price * fee_bps / 10000
    position['cash'] -= fee
    position['fees'] += fee
    position['fills'].append(dict(time=int(at), signed_qty=float(signed_qty), price=float(price), fee=fee, source=source))


def fresh_position(symbol, decimals):
    return dict(symbol=symbol, decimals=decimals, units=0, qty=0., cash=0., fees=0., funding=0., fills=[])


def audit_probe(probe, streams, decimals, d, symbols, cfg, sc):
    fill=probe['lead_fill']; lead=probe['lead']; other=probe['hedge']; beta=probe['beta']
    positions=[fresh_position(lead,decimals[lead]), fresh_position(other,decimals[other])]
    apply_fill(positions[0],fill['time'],fill['side']*fill['qty'],fill['price'],sc['maker_fee_bps'],'hypothetical_maker_queue')
    sn=asof(streams[lead],fill['time'],cfg); hn=asof(streams[other],fill['time'],cfg)
    hedge_status='missing_hedge_decision'; hedge_time=fill['time']; requested=0.; got=0.
    if sn is not None and hn is not None:
        multiplier=beta if probe['pair'][0]==lead else 1/beta
        ideal=fill['qty']*probe['hedge_ratio_units']
        requested=lot_floor(ideal,decimals[other])
        side=-fill['side']; top=float(hn['book']['levels'][1 if side>0 else 0][0]['px'])
        limit=probe['hedge_limit']
        if requested*top<cfg['minimum_hedge_notional'] or requested/ideal<.95:
            hedge_status='hedge_below_minimum'
        else:
            arrival=delayed_snapshot(streams[other],fill['time']+cfg['entry_latency_ms'],cfg)
            if arrival is None:
                hedge_status='missing_hedge_arrival'
                # An absent observation cannot establish a zero fill. This
                # path is an explicitly assumed unfilled-hedge scenario and
                # remains flagged as an unknown historical order outcome.
                hedge_time=fill['time']+cfg['entry_latency_ms']+cfg['future_exit_observation_grace_ms']
            else:
                hedge_time, hs=arrival
                got, px=sweep(hs['book']['levels'][1 if side>0 else 0],requested,sc['depth_fraction'],side,decimals[other],limit)
                if got:
                    apply_fill(positions[1],hedge_time,side*got,px,sc['taker_fee_bps'],'observed_l2_hedge')
                hedge_status='full' if got>=requested-1e-10 else 'partial' if got else 'unfilled_within_limit'
    balanced=requested>0 and got/requested>=.95
    exit_decision=probe['forecast_maturity_ms'] if balanced else hedge_time
    deadline=min(exit_decision+cfg['maximum_exit_retry_hours']*3600000,int(d['times'][-1]*1000))
    cutoff=int(np.searchsorted(d['times']*1000,deadline,side='right')-1)
    assert cutoff>=0
    # Funding is reconstructed below from chronological fills, avoiding a
    # payment on a quantity which had not yet entered at settlement time.
    l2_exit_status=[]
    for pos in positions:
        if not pos['qty']: continue
        symbol=pos['symbol']; side=-int(np.sign(pos['qty']))
        snap=asof(streams[symbol],exit_decision,cfg)
        arrival=delayed_snapshot(streams[symbol],exit_decision+cfg['entry_latency_ms'],cfg) if snap is not None else None
        status='unobserved'
        if arrival is not None and arrival[0]<=deadline:
            at,ss=arrival; top=float(snap['book']['levels'][1 if side>0 else 0][0]['px'])
            limit=top*(1+side*cfg['hedge_limit_bps']/10000)
            closed,px=sweep(ss['book']['levels'][1 if side>0 else 0],abs(pos['qty']),sc['depth_fraction'],side,decimals[symbol],limit)
            if closed:apply_fill(pos,at,side*closed,px,sc['taker_fee_bps'],'observed_l2_exit')
            status='closed' if abs(pos['qty'])<1e-10 else 'partial' if closed else 'unfilled'
        l2_exit_status.append(dict(symbol=symbol,status=status))
        # The entire next bar is unknown at decision time. Execution at its
        # close plus impact is a model; capacity is realized during that bar.
        first=int(np.searchsorted(d['times']*1000,max(exit_decision, pos['fills'][-1]['time']),side='right'))
        j=symbols.index(symbol)
        for i in range(first,cutoff+1):
            if abs(pos['qty'])<1e-10:break
            volume=float(d['par_volume'][i,j])
            if volume<=0:continue
            size=lot_floor(min(abs(pos['qty']),volume*sc['candle_participation']),decimals[symbol])
            if not size:continue
            price=float(d['par_close'][i,j]*(1+side*sc['exit_impact_bps']/10000))
            apply_fill(pos,int(d['times'][i]*1000),side*size,price,sc['taker_fee_bps'],'modeled_candle_exit')
        # Numerical dust below half a lot can only be floating-point residue:
        # all entry and reduction quantities lie on the same size grid.
        if abs(pos['qty'])<.5*10**(-decimals[symbol]):pos['qty']=0.
    funding_rows=[]; cash_pnl=0.; gross_pnl=0.; fees=0.; residual=0.; fees_entry=0.; notional=0.
    for pos in positions:
        j=symbols.index(pos['symbol']);mark=float(d['par_close'][cutoff,j]);flows=sorted(pos['fills'],key=lambda f:f['time'])
        for i in np.flatnonzero((d['times']%3600==0)&(d['times']*1000>fill['time'])&(d['times']*1000<=deadline)):
            at=int(d['times'][i]*1000)
            quantity=sum(f['signed_qty'] for f in flows if f['time']<at)
            if abs(quantity)<1e-10:continue
            assert d['funding_observed'][i,j]
            cash=-quantity*float(d['reference_close'][i,j])*float(d['funding'][i,j])
            if cash>0:cash*=sc['funding_receipt_multiplier']
            pos['funding']+=cash
            funding_rows.append(dict(time=at,symbol=pos['symbol'],qty=quantity,amount=cash,oracle_proxy=True))
        pos['mark']=mark;pos['marked_pnl']=pos['cash']+pos['qty']*mark+pos['funding']
        pos['price_pnl']=sum(f['signed_qty']*(mark-f['price']) for f in flows)
        cash_pnl+=pos['marked_pnl'];gross_pnl+=pos['price_pnl'];fees+=pos['fees'];residual+=abs(pos['qty'])*mark
        for f in flows:
            if f['source'] in ('hypothetical_maker_queue','observed_l2_hedge'):
                notional+=abs(f['signed_qty'])*f['price'];fees_entry+=f['fee']
    funding=sum(p['funding'] for p in positions)
    error=abs(cash_pnl-(gross_pnl+funding-fees));assert error<1e-7
    reserve=residual*(sc['taker_fee_bps']+sc['exit_impact_bps'])/10000
    close_times=[f['time'] for p in positions for f in p['fills'] if f['source'].endswith('_exit')]
    closed=not any(p['qty'] for p in positions)
    return dict(pair=probe['pair'],lead=lead,hours=probe['hours'],tape=probe['tape'],phase=probe['phase'],
        quote_decision=probe['quote_decision'],forecast_aligned=probe['forecast_aligned'],lead_fill=fill,
        original_status=probe['status'],hedge_status=hedge_status,requested_hedge=requested,filled_hedge=got,
        balanced=balanced,exit_decision=exit_decision,deadline=deadline,l2_exit_status=l2_exit_status,
        unobserved_hedge_outcome=hedge_status=='missing_hedge_arrival',
        status='closed_observed_or_modeled' if closed else 'unresolved_inventory',positions=positions,
        net_pnl=cash_pnl,net_after_reserve=cash_pnl-reserve,notional=notional,fees=fees,entry_fees=fees_entry,
        funding=funding,price_pnl=gross_pnl,residual_gross=residual,liquidation_reserve=reserve,
        exit_delay_minutes=(max(close_times)-exit_decision)/60000 if close_times else None,
        funding_events=funding_rows,accounting_error=error,not_account_return_or_Q=True)

