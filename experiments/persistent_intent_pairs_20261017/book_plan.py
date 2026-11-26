"""Read-only entry planner from a frozen price hypothesis and current books.

It returns bounded independent orders, never an atomic pair fill assumption.
No live exchange client or signer is imported.
"""
import numpy as np
from experiments.basis_carry_20261017.probe import walk
from replay_lab.common_comparison.engine import rounded,risk_breaches


def plan_entry(intent,books,now,cfg,p,sc,qty,equity,sigma):
    def reject(reason):return dict(accepted=False,reason=reason,orders=[])
    source=int(intent['forecast_source']);age=(now-source)/3600
    if age<0:return reject('future_signal')
    if age>=cfg['book_intent_lifetime_hours']:return reject('expired_signal')
    ids=intent['ids'];side=int(intent['direction']);beta=float(intent['beta'])
    if not .1<=beta<=10 or side not in (-1,1):return reject('invalid_hedge')
    if np.any(np.asarray(qty)[ids]):return reject('existing_pair_inventory')
    # The caller supplies all-account marks only for global risk; never use
    # candle volume to decide whether a new IOC can consume visible liquidity.
    prices=np.asarray(intent['account_marks'],float).copy();quotes=[]
    for symbol,j in zip(intent['pair'],ids):
        book=books.get(symbol)
        if book is None:return reject('unobserved_book')
        received=book['received'];clock=now*1000
        if not 0<=clock-received['book']<=cfg['book_max_age_ms']:return reject('stale_or_future_book')
        if not 0<=clock-received['ctx']<=cfg['context_max_age_ms']:return reject('stale_or_future_context')
        if not -1000<=clock-book['book']['time']<=cfg['book_max_age_ms']:return reject('stale_server_book')
        levels=book['book']['levels']
        if not all(levels):return reject('empty_book')
        bid=float(levels[0][0]['px']);ask=float(levels[1][0]['px'])
        if bid<=0 or ask<bid:return reject('invalid_book')
        prices[j]=(bid+ask)/2;quotes.append(book)
    z=(np.log(prices[ids[0]])-beta*np.log(prices[ids[1]])-intent['entry_mean'])/intent['entry_sd']
    if not -cfg['entry_z']*cfg['stop_multiplier']<side*z<0:return reject('mean_or_stop_reached')
    weights=side*np.array([1.,-beta])/(1+beta);caps=[]
    for w,quote in zip(weights,quotes):
        levels=quote['book']['levels'][1 if w>0 else 0];best=float(levels[0]['px'])
        limit=best*(1+np.sign(w)*cfg['visible_depth_band_bps']/10000)
        caps.append(sum(float(l['sz'])*sc['depth_fraction'] for l in levels if np.sign(w)*(float(l['px'])-limit)<=1e-10))
    gross=min(cfg['pair_max_gross'],min(np.asarray(caps)*prices[ids]/abs(weights)))
    def feasible(g):
        for mask in ([1,0],[0,1],[1,1]):
            proposal=np.asarray(qty,float).copy();proposal[ids]+=g*weights/prices[ids]*mask
            if risk_breaches(proposal,equity-g*.01,prices,sigma,p['risk']):return False
        return True
    if not feasible(0):return reject('existing_risk_breach')
    if not feasible(gross):
        lo,hi=0.,gross
        for _ in range(40):
            mid=(lo+hi)/2
            if feasible(mid):lo=mid
            else:hi=mid
        gross=lo
    target=np.array([rounded(gross*w/prices[j],p['execution']['decimals'][j]) for j,w in zip(ids,weights)])
    if np.any(abs(target)*prices[ids]<p['execution']['minimum_order_value']):return reject('visible_depth_below_minimum')
    actual_beta=abs(target[1])*prices[ids[1]]/(abs(target[0])*prices[ids[0]])
    if abs(actual_beta/beta-1)>cfg['pair_balance_tolerance']:return reject('lot_imbalance')
    entries=[];exits=[];oracles=[];funding=[]
    for q,quote in zip(target,quotes):
        entry=walk(quote['book']['levels'][1 if q>0 else 0],abs(q),sc['depth_fraction'])
        exit_=walk(quote['book']['levels'][0 if q>0 else 1],abs(q),sc['depth_fraction'])
        if entry is None or exit_ is None:return reject('visible_entry_or_exit_depth_missing')
        entries.append(entry);exits.append(exit_);oracles.append(quote['oracle']);funding.append(float(quote['ctx']['funding']))
    entries=np.asarray(entries);exits=np.asarray(exits);terminal=np.asarray(intent['terminal_prices'])
    if not np.isfinite(terminal).all() or np.any(terminal<=0):return reject('invalid_terminal_hypothesis')
    gross=float(abs(target)@prices[ids]);fee=sc['fee_bps']/10000
    remaining=max(0.,cfg['funding_forecast_hours']-age)
    cash=-target*np.asarray(oracles)*np.asarray(funding)*remaining
    cash=float(np.where(cash>0,cash*.5,cash).sum())
    # Current reverse-side depth is an exit-cost proxy, with an extra adverse
    # move reserve. It is not a promise that future exit liquidity will exist.
    exit_cost=float(target@(prices[ids]-exits))+gross*(fee+sc['exit_impact_bps']/10000)
    buffer=gross*cfg['entry_buffer_bps']/10000
    def expected(px):return float(target@(terminal-px)-abs(target)@px*fee+cash-exit_cost)
    net=expected(entries);slack=net-buffer
    if not np.isfinite(net) or slack<=0:return reject('edge_below_full_cost')
    allocations=abs(target)*prices[ids]/gross
    distance=slack*allocations/(abs(target)*(1+np.sign(target)*fee))
    distance=np.minimum(distance,entries*cfg['entry_limit_extra_bps']/10000)
    limits=entries+np.sign(target)*distance
    if np.any(limits<=0):return reject('invalid_limit')
    worst=expected(limits);assert worst>=buffer-1e-8
    # Verify the sealed limits against account constraints, including either
    # leg filling alone; do not rely only on the 1% preliminary haircut.
    for mask in ([1,0],[0,1],[1,1]):
        change=target*np.asarray(mask);q=np.asarray(qty,float).copy();q[ids]+=change
        marked=equity-float(change@(limits-prices[ids]))-float(abs(change)@limits)*fee
        if marked<=0 or risk_breaches(q,marked,prices,sigma,p['risk']):return reject('risk_at_sealed_limits')
    orders=[dict(symbol=s,asset=j,signed_qty=float(q),limit=float(limit),time_in_force='Ioc')
        for s,j,q,limit in zip(intent['pair'],ids,target,limits)]
    return dict(accepted=True,reason='bounded_pair_intent',orders=orders,decision_time=now,forecast_source=source,
        target=target.tolist(),entry_preview=entries.tolist(),exit_preview=exits.tolist(),
        expected_net_cash=net,minimum_expected_cash_at_limits=worst,buffer_cash=buffer,
        expected_funding_cash=cash,exit_cost_reserve=exit_cost,terminal_prices=terminal.tolist(),
        requires_independent_fill_reconciliation=True,future_exit_liquidity_guaranteed=False,orders_sent=0)
