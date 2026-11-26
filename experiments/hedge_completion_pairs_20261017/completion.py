"""One missing leg may be completed using a frozen episode cash budget."""
import numpy as np
from replay_lab.common_comparison.engine import rounded,risk_breaches
from replay_lab.pullback.broker import execution_price
from experiments.adaptive_mean_pairs_20261017.signals import held_state


def completion_plan(pair,ledger,i,d,de,cfg,p,sc):
    now=int(d['times'][i]);ids=pair['ids'];held=ledger.qty[ids].copy();want=np.asarray(pair['target'])
    if now>=pair['completion_deadline'] or np.count_nonzero(held)!=1:return None
    if held_state(pair,i,d,de,cfg)['reason'] is not None:return None
    have=int(np.flatnonzero(held)[0]);missing=1-have
    ratio=held[have]/want[have]
    if ratio<=0 or ratio>1+1e-8:return None
    target=want*ratio;target[have]=held[have]
    target[missing]=rounded(target[missing],p['execution']['decimals'][ids[missing]])
    if target[missing]==0:return None
    ratios=target/want
    if abs(ratios[0]-ratios[1])/max(ratios)>cfg['pair_balance_tolerance']:return None
    prices=d['par_close'][i,ids];gross=float(abs(target)@prices)
    j=ids[missing];qty=target[missing];side=np.sign(qty)
    preview=execution_price(prices[missing],side,sc['impact_bps'],p['execution']['decimals'][j])
    if abs(qty)*preview<p['execution']['minimum_order_value']:return None
    all_qty=ledger.qty.copy();all_qty[ids]=target
    marked=ledger.equity(d['par_close'][i])-abs(qty)*preview*sc['fee_bps']/10000-qty*(preview-prices[missing])
    if risk_breaches(all_qty,marked,d['par_close'][i],d['sigma'][i],p['risk']):return None
    # Only the pair's own fills/funding enter this episode, including in overlay accounts.
    sunk_fees=sum(f['fee'] for f in ledger.fills if f['time']>pair['intent_time'] and f['asset'] in ids)
    realized_carry=0.
    for f in ledger.funding_events:
        if f['time']<=pair['intent_time']:continue
        realized_carry-=float(np.asarray(f['qty'])[ids]@(np.asarray(f['prices'])[ids]*np.asarray(f['rates'])[ids]))
    remaining=max(0.,cfg['funding_forecast_hours']-(now-pair['forecast_source'])/3600)
    projected_carry=-target*d['reference_close'][i,ids]*d['known_hourly_funding'][i,ids]*remaining
    projected_carry=float(np.where(projected_carry>0,.5*projected_carry,projected_carry).sum())
    entries=np.array([ledger.entry[x] for x in ids]);entries[missing]=preview
    terminal=np.asarray(pair['terminal_prices'])
    exit_reserve=gross*(sc['fee_bps']+sc['impact_bps'])/10000
    buffer=gross*cfg['entry_buffer_bps']/10000
    def expected(price):
        entry=entries.copy();entry[missing]=price
        return float(target@(terminal-entry)-sunk_fees-abs(qty)*price*sc['fee_bps']/10000+
                     realized_carry+projected_carry-exit_reserve)
    net=expected(preview);slack=net-buffer
    if not np.isfinite(net) or slack<=0:return None
    distance=slack/(abs(qty)*(1+side*sc['fee_bps']/10000))
    limit=preview+side*distance
    if limit<=0:return None
    worst=expected(limit)
    assert worst>=buffer-1e-8
    return dict(target=target.tolist(),missing_asset=j,entry_limits={str(j):float(limit)},
        preview_price=float(preview),expected_net_cash=net,minimum_expected_cash_at_limits=worst,
        buffer_cash=buffer,sunk_pair_fees=float(sunk_fees),realized_pair_funding=realized_carry,
        expected_future_funding=projected_carry,exit_reserve=exit_reserve,
        frozen_terminal_prices=terminal.tolist(),deadline=pair['completion_deadline'])


def after_completion(pair,all_qty,p,cfg,now):
    pair['completing']=False;q=all_qty[pair['ids']];want=np.asarray(pair['target']);ratios=q/want
    if min(ratios)>0 and abs(ratios[0]-ratios[1])/max(ratios)<=cfg['pair_balance_tolerance']:
        pair['balanced']=True
        return 'balanced'
    if min(ratios)>0:
        trim=np.array([rounded(x*min(ratios),p['execution']['decimals'][j]) for x,j in zip(want,pair['ids'])])
        if np.all(trim!=0):
            pair.update(repairing=True,repair_deadline=now+cfg['repair_deadline_seconds'],repair_qty=trim.tolist())
            return 'trim_requested'
    pair.update(closing=True,aborted=True)
    return 'abort'
